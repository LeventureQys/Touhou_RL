"""Vision student (D10–D12 skeleton): frame image → action, distilled offline.

Design
------
* input: two stacked grayscale frame images (``vision.render_state``), hazards 1.0, player
  0.5 — the same representation the live front-end will have to produce;
* model: a small CNN (two conv/pool blocks + linear head) predicting the
  teacher's 18-way action;
* teacher: the D5 controller's chosen action at the same decision;
* evaluation: (a) held-out agreement with the teacher, (b) a closed loop where
  the student's argmax is executed in the synthetic world.

Honesty boundary
----------------
The images are *synthetic* (rasterised from privileged state), not the game's
frame buffer.  Agreement on these images says the student can fit this
rasterisation; it says nothing about transfer to real pixels.  Every artefact
carries ``synthetic_frames: true`` and the report repeats it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import controller, planner, synth, vision

TEACHER_SCHEMA = "v370-student/1"


@dataclass(frozen=True)
class StudentConfig:
    episodes: int = 12
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 7
    device: str = "auto"
    #: 1 = single frame, 2 = current + previous frame.  Measured A/B (same split,
    #: 20 episodes, 16 epochs): 1 channel 0.614 test accuracy, 2 channels 0.523.
    #: The two-frame stack did not pay off at this data scale, so 1 is the default.
    channels: int = 1
    #: Append the eight "future safety by direction" scalars from ``vision``.
    future_features: bool = False
    #: Auxiliary head: predict the 8×8 hazard occupancy ``horizon`` frames ahead.
    #: This forces the trunk to represent the future instead of receiving it.
    aux_future_map: bool = False
    aux_weight: float = 0.5
    #: Learn the *route* (normalised endpoint direction) instead of the action
    #: label.  The four input/structure experiments showed the action-cloning
    #: ceiling is set by "the label is a function of the route"; this tests the
    #: obvious alternative.
    route_target: bool = False
    route_weight: float = 1.0
    #: Force bit-reproducible training.  CUDA `randperm` and cuDNN convolutions are
    #: not deterministic by default: the same config measured 19/23 and 18/23 in
    #: two runs, which is larger than the A/B effect we were trying to read.
    deterministic: bool = True
    planner: controller.ControllerConfig = field(default_factory=controller.ControllerConfig)


def _scenarios(seed: int, episodes: int) -> list[synth.Scenario]:
    from . import families

    edges = families.edge_instances()
    ordered = [edges[key] for key in sorted(edges)]
    out = []
    families_order = ("SC-01", "SC-02", "SC-03", "SC-08", "SC-09", "SC-15", "SC-16", "SC-17")
    for index in range(episodes):
        if index < len(ordered):
            out.append(ordered[index])
        else:
            family = families_order[(index - len(ordered)) % len(families_order)]
            out.append(families.build(family, (index * 7 + seed) % 30))
    return out


def collect_teacher(config: StudentConfig | None = None) -> dict:
    """Run the controller and store (image, action) pairs from privileged state."""

    config = config or StudentConfig()
    images: list[np.ndarray] = []
    labels: list[int] = []
    episode_ids: list[str] = []
    extras: list[np.ndarray] = []
    future_maps: list[np.ndarray] = []
    route_targets: list[np.ndarray] = []
    for scenario in _scenarios(config.seed, config.episodes):
        env = synth.SyntheticEnv(scenario)
        from . import forecast

        field = forecast.DangerField.from_scenario(scenario, 0)
        ctrl = controller.EscapeController(config.planner)
        limit = scenario.horizon_frames // config.planner.planner.frameskip
        previous_position: tuple[float, float] | None = None
        for _ in range(limit):
            if env.hit or env.frame >= scenario.horizon_frames:
                break
            decision, _diag = ctrl.decide(
                x=env.x, y=env.y, frame=env.frame,
                previous_action=env.previous_action, field=field, bombs=0,
            )
            if config.channels == 1:
                image = vision.render_state(scenario, env.frame, env.x, env.y)[None]
            else:
                image = vision.render_pair(
                    scenario,
                    env.frame,
                    previous_position,
                    (env.x, env.y),
                )
            images.append(image)
            if config.future_features:
                extras.append(vision.future_features(scenario, env.frame, env.x, env.y))
            if config.aux_future_map:
                future_maps.append(vision.future_map(scenario, env.frame, env.x, env.y))
            if config.route_target:
                plan = ctrl.state.plan
                if plan is not None and plan.record.position_time_knots:
                    end = plan.record.position_time_knots[-1]
                    route_targets.append(
                        np.array([(end.x - env.x) / 100.0, (end.y - env.y) / 100.0], dtype=np.float32)
                    )
                else:
                    route_targets.append(np.zeros(2, dtype=np.float32))
            action = (int(decision.accepted_action[2]), int(decision.accepted_action[0]))
            labels.append(planner.ACTIONS.index(action))
            episode_ids.append(scenario.scenario_id)
            previous_position = (env.x, env.y)
            env.step(action)
    return {
        "images": np.stack(images) if images else np.zeros((0, config.channels) + vision.IMAGE_SHAPE, np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "extras": np.stack(extras) if extras else np.zeros((0, 8), np.float32),
        "future_maps": np.stack(future_maps) if future_maps else np.zeros((0, 8, 8), np.float32),
        "route_targets": np.stack(route_targets) if route_targets else np.zeros((0, 2), np.float32),
        "episodes": episode_ids,
        "scenarios": sorted(set(episode_ids)),
    }


class StudentNet:
    """Tiny CNN (plus an optional 8-dim future-feature branch)."""

    def __init__(
        self,
        device: str = "auto",
        seed: int = 7,
        channels: int = 2,
        extra_dim: int = 0,
        aux_map: bool = False,
        route_head: bool = False,
    ):
        import torch
        from torch import nn

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)
        self.device = device
        self.extra_dim = int(extra_dim)
        trunk_out = 16 * (vision.IMAGE_SHAPE[0] // 4) * (vision.IMAGE_SHAPE[1] // 4)
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        ).to(device)
        self.head = nn.Linear(trunk_out + self.extra_dim, 18).to(device)
        self.aux_map = bool(aux_map)
        self.map_head = nn.Linear(trunk_out, 64).to(device) if self.aux_map else None
        self.route_head = nn.Linear(trunk_out, 2).to(device) if route_head else None

    def forward(self, x, extra=None):
        features = self.trunk(x)
        if self.extra_dim:
            if extra is None:
                raise ValueError("model expects the 8-dim future-feature branch")
            features = __import__("torch").cat([features, extra], dim=1)
        return self.head(features)

    def fit(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        config: StudentConfig,
        extras: np.ndarray | None = None,
        maps: np.ndarray | None = None,
        routes: np.ndarray | None = None,
    ) -> list[float]:
        import torch
        from torch import nn

        if config.deterministic:
            torch.manual_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        generator = torch.Generator(device=self.device).manual_seed(config.seed)
        x = torch.from_numpy(images).float().to(self.device)
        y = torch.from_numpy(labels).long().to(self.device)
        e = None
        if self.extra_dim:
            if extras is None or extras.size == 0:
                raise ValueError("missing future features for a model that needs them")
            e = torch.from_numpy(extras).float().to(self.device)
        m = None
        if self.aux_map:
            if maps is None or maps.size == 0:
                raise ValueError("missing future maps for a model with the auxiliary head")
            m = torch.from_numpy(maps).float().to(self.device)
        r = None
        if self.route_head is not None:
            if routes is None or routes.size == 0:
                raise ValueError("missing route targets for a model with the route head")
            r = torch.from_numpy(routes).float().to(self.device)
        parameters = list(self.trunk.parameters()) + list(self.head.parameters())
        if self.map_head is not None:
            parameters += list(self.map_head.parameters())
        if self.route_head is not None:
            parameters += list(self.route_head.parameters())
        optimiser = torch.optim.Adam(parameters, lr=config.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        map_loss_fn = nn.BCEWithLogitsLoss()
        route_loss_fn = nn.MSELoss()
        losses: list[float] = []
        for _epoch in range(config.epochs):
            permutation = torch.randperm(x.shape[0], device=self.device, generator=generator)
            epoch_loss = 0.0
            for start in range(0, x.shape[0], config.batch_size):
                index = permutation[start : start + config.batch_size]
                optimiser.zero_grad()
                batch_extra = e[index] if e is not None else None
                features = self.trunk(x[index])
                if batch_extra is not None:
                    features = torch.cat([features, batch_extra], dim=1)
                logits = self.head(features)
                loss = loss_fn(logits, y[index])
                if self.map_head is not None and m is not None:
                    predicted = self.map_head(self.trunk(x[index])).reshape(-1, 8, 8)
                    loss = loss + config.aux_weight * map_loss_fn(predicted, m[index])
                if self.route_head is not None and r is not None:
                    route_pred = torch.tanh(self.route_head(self.trunk(x[index])))
                    loss = loss + config.route_weight * route_loss_fn(route_pred, r[index])
                loss.backward()
                optimiser.step()
                epoch_loss += float(loss.detach()) * index.numel()
            losses.append(round(epoch_loss / max(1, x.shape[0]), 6))
        return losses

    def predict_route(self, images: np.ndarray) -> np.ndarray:
        import torch

        if self.route_head is None:
            raise ValueError("model has no route head")
        with torch.no_grad():
            x = torch.from_numpy(images).float()
            if x.ndim == 3:
                x = x.unsqueeze(0)
            x = x.to(self.device)
            return torch.tanh(self.route_head(self.trunk(x))).cpu().numpy()

    def predict(self, images: np.ndarray, extras: np.ndarray | None = None) -> np.ndarray:
        import torch

        with torch.no_grad():
            x = torch.from_numpy(images).float().to(self.device)
            # (C, H, W) is treated by torch as an *unbatched* input, and Flatten(1)
            # then keeps the channel dim; always add the batch dim explicitly.
            if x.ndim == 3:
                x = x.unsqueeze(0)
            e = None
            if self.extra_dim:
                if extras is None:
                    raise ValueError("model expects the 8-dim future-feature branch")
                e = torch.from_numpy(np.asarray(extras)).float().to(self.device)
                if e.ndim == 1:
                    e = e.unsqueeze(0)
            return self.forward(x, e).argmax(dim=1).cpu().numpy()


def evaluate_student(
    model: StudentNet,
    images: np.ndarray,
    labels: np.ndarray,
    extras: np.ndarray | None = None,
) -> dict:
    predictions = model.predict(images, extras)
    accuracy = float((predictions == labels).mean()) if labels.size else None
    confusion: dict[str, int] = {}
    for truth, predicted in zip(labels.tolist(), predictions.tolist()):
        key = f"{truth}->{predicted}"
        confusion[key] = confusion.get(key, 0) + 1
    top = dict(sorted(confusion.items(), key=lambda item: item[1], reverse=True)[:8])
    # Escape behaviour is carried by the *move* component; the focus bit is a
    # detail.  Reporting both keeps a 0.6 exact accuracy from looking like a
    # failure when every direction is right.
    move_truth = labels % 9
    move_pred = predictions % 9
    focus_truth = labels // 9
    focus_pred = predictions // 9
    return {
        "samples": int(labels.size),
        "accuracy": None if accuracy is None else round(accuracy, 4),
        "move_accuracy": None if labels.size == 0 else round(float((move_truth == move_pred).mean()), 4),
        "focus_accuracy": None if labels.size == 0 else round(float((focus_truth == focus_pred).mean()), 4),
        "chance": round(1.0 / 18.0, 4),
        "move_chance": round(1.0 / 9.0, 4),
        "confusion_top": top,
    }


def _action_towards(direction: np.ndarray) -> tuple[int, tuple[int, int]]:
    """Pick the move whose one-frame displacement best matches ``direction``."""

    model_geometry = __import__("touhou_rl.escape.geometry", fromlist=["MoveModel"]).MoveModel()
    best_index = 0
    best_score = -2.0
    norm = float(np.hypot(direction[0], direction[1])) or 1.0
    unit = np.asarray(direction, dtype=np.float64) / norm
    for index, action in enumerate(planner.ACTIONS):
        velocity = model_geometry.displacement(action[0], action[1], frames=1)
        speed = float(np.hypot(velocity[0], velocity[1])) or 1.0
        score = (velocity[0] * unit[0] + velocity[1] * unit[1]) / speed
        if score > best_score + 1e-9:
            best_score = score
            best_index = index
    return best_index, planner.ACTIONS[best_index]


def closed_loop_student(
    model: StudentNet,
    scenario: synth.Scenario,
    *,
    max_decisions: int | None = None,
    channels: int = 2,
    route_policy: bool = False,
) -> dict:
    """Execute the student in the synthetic world (no teacher involved).

    ``route_policy`` uses the route head: predict the endpoint direction, then pick
    the move that best matches it.  This is the "learn the route, not the action"
    alternative to action cloning.
    """

    env = synth.SyntheticEnv(scenario)
    limit = max_decisions or scenario.horizon_frames // 2
    intents: dict[int, int] = {}
    previous_xy: tuple[float, float] | None = None
    for _ in range(limit):
        if env.hit or env.frame >= scenario.horizon_frames:
            break
        if channels == 1:
            image = vision.render_state(scenario, env.frame, env.x, env.y)[None]
        else:
            image = vision.render_pair(scenario, env.frame, previous_xy, (env.x, env.y))[None]
        extra = None
        if model.extra_dim:
            extra = vision.future_features(scenario, env.frame, env.x, env.y)[None]
        if route_policy:
            direction = model.predict_route(image)[0]
            action_index, action = _action_towards(direction)
        else:
            action_index = int(model.predict(image, extra)[0])
            action = planner.ACTIONS[action_index]
        intents[action_index] = intents.get(action_index, 0) + 1
        previous_xy = (env.x, env.y)
        env.step(action)
    return {
        "scenario": scenario.scenario_id,
        "frames": env.frame,
        "hit": env.hit,
        "survived": (not env.hit) and env.frame >= scenario.horizon_frames,
        "min_clearance": None if env.min_clearance == float("inf") else round(env.min_clearance, 4),
        "action_histogram": dict(sorted(intents.items())),
    }


def train_and_evaluate(config: StudentConfig | None = None, *, holdout: float = 0.2) -> dict:
    config = config or StudentConfig()
    dataset = collect_teacher(config)
    images = dataset["images"]
    labels = dataset["labels"]
    if labels.size == 0:
        raise RuntimeError("no teacher samples collected")
    rng = np.random.default_rng(config.seed)
    # Split by *scenario*, not by sample: adjacent frames of the same episode are
    # almost identical, so a sample-wise split leaks the answer into the test set.
    episode_ids = np.asarray(dataset["episodes"])
    unique = sorted(set(episode_ids.tolist()))
    rng.shuffle(unique)
    test_count = max(1, int(round(len(unique) * holdout))) if len(unique) > 1 else 0
    test_episodes = set(unique[:test_count])
    train_index = np.asarray([i for i, item in enumerate(episode_ids) if item not in test_episodes])
    test_index = np.asarray([i for i, item in enumerate(episode_ids) if item in test_episodes])
    if test_index.size == 0:
        test_index = train_index
    model = StudentNet(
        config.device, config.seed, config.channels,
        8 if config.future_features else 0,
        aux_map=config.aux_future_map,
        route_head=config.route_target,
    )
    extras = dataset["extras"] if config.future_features else None
    maps = dataset["future_maps"] if config.aux_future_map else None
    routes = dataset["route_targets"] if config.route_target else None
    losses = model.fit(
        images[train_index], labels[train_index], config,
        extras[train_index] if extras is not None else None,
        maps[train_index] if maps is not None else None,
        routes[train_index] if routes is not None else None,
    )
    train_metrics = evaluate_student(model, images[train_index], labels[train_index])
    test_metrics = evaluate_student(
        model, images[test_index], labels[test_index],
        extras[test_index] if extras is not None else None,
    )
    from . import families

    loop_scenarios = [families.edge_instances()["E-01"], families.build("SC-17", 0)]
    loops = [
        closed_loop_student(model, scenario, channels=config.channels, route_policy=config.route_target)
        for scenario in loop_scenarios
    ]
    return {
        "schema": TEACHER_SCHEMA,
        "synthetic_frames": True,
        "note": "images are rasterised from privileged state, not the game frame buffer",
        "config": {
            "episodes": config.episodes,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "device": config.device,
        },
        "samples": int(labels.size),
        "episodes": len(dataset["scenarios"]),
        "split": {
            "by": "scenario",
            "train_episodes": len(set(episode_ids[train_index].tolist())),
            "test_episodes": len(set(episode_ids[test_index].tolist())),
            "train_samples": int(train_index.size),
            "test_samples": int(test_index.size),
        },
        "losses": losses,
        "train": train_metrics,
        "test": test_metrics,
        "closed_loop": loops,
    }
