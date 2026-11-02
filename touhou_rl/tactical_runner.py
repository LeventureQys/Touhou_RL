"""Transactional single-environment runner; never sends inputs or assumes execution.

One pending proposal at a time. commit is the execution acknowledgement; cancel
rolls back the speculative hidden state. Death, respawn, frame rollback or input
exceptions must call reset (on_exception/on_death aliases provided). No durations
or action chunks are predicted: duration is always one base control period.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .tactical_policy import bos_history


@dataclass(frozen=True)
class TacticalProposal:
    proposal_id: int
    candidate: int
    option: int
    target: tuple[float, float]
    duration: int = 1


class TacticalRunner:
    def __init__(self, policy):
        self.policy = policy
        self._counter = 0
        self.reset()

    def reset(self):
        self.hidden = None
        self.previous_candidate = None
        self.previous_hold = None
        self.previous_option = None
        self.previous_target = None
        self.target_valid = False
        self.effective_frame = None
        self._pending = None

    on_exception = reset
    on_death = reset

    def cancel(self, proposal_id=None):
        if self._pending is None:
            raise RuntimeError("no pending proposal")
        if proposal_id is not None and proposal_id != self._pending[0].proposal_id:
            raise ValueError("stale proposal id")
        self._pending = None

    def _history(self, features, observation_frame):
        history = bos_history(features)
        if self.previous_candidate is None:
            return history
        if observation_frame is None:
            raise ValueError("observation_frame required after first commit")
        if type(observation_frame) is not int or observation_frame < self.effective_frame:
            raise ValueError("observation frame rollback: reset required")
        history["previous_candidate"].fill_(self.previous_candidate)
        history["previous_option"].fill_(self.previous_option)
        history["previous_hold"].fill_(self.previous_hold)
        history["frame_delta"].fill_(observation_frame-self.effective_frame)
        history["previous_target"][0,0] = features.new_tensor(self.previous_target)
        history["target_valid"].fill_(self.target_valid)
        history["history_valid"].fill_(True)
        return history

    @torch.no_grad()
    def propose(self, regions=None, *, features=None, base_logits=None, observation_frame=None):
        if self._pending is not None:
            raise RuntimeError("commit or cancel the pending proposal first")
        if observation_frame is not None and (type(observation_frame) is not int or observation_frame < 0):
            raise ValueError("observation_frame must be a nonnegative integer")
        self.policy.eval()
        if regions is not None:
            if features is not None or base_logits is not None:
                raise ValueError("provide pixels OR cached visual inputs")
            core = self.policy.vision_model.forward_core(regions)
            features, base_logits = core["features"], core["policy_logits"]
        if features is None or base_logits is None:
            raise ValueError("pixels or features/base_logits required")
        if features.ndim == 2:
            features = features[:,None]
        if base_logits.ndim == 2:
            base_logits = base_logits[:,None]
        if features.shape[:2] != (1,1) or base_logits.shape != (1,1,18):
            raise ValueError("runner supports exactly one environment and one observed step")
        history = self._history(features, observation_frame)
        output = self.policy.forward_features(features, base_logits, history, self.hidden)
        self._counter += 1
        proposal = TacticalProposal(self._counter, int(output["policy_logits"][0,0].argmax()),
                                    int(output["option_logits"][0,0].argmax()),
                                    tuple(output["target"][0,0].tolist()))
        # Clone so caller mutation of output cannot change the commit transaction.
        self._pending = (proposal, output["hidden"].detach().clone(), observation_frame)
        return proposal

    def commit(self, actual_action, actual_hold, effective_frame, *, actual_option=None,
               target_adopted=True, proposal_id=None):
        """Acknowledge *actual* execution, including overrides of proposed action.

        effective_frame is the action's actual input-effective start frame;
        actual_hold is measured logical-frame exposure (positive integer).
        actual_option defaults to adopting the proposal's option; external
        fallback can pass BOS=3 and target_adopted=False. No unexecuted option
        or target should be acknowledged as adopted.
        """
        if self._pending is None:
            raise RuntimeError("commit requires a pending proposal")
        proposal, hidden, observed = self._pending
        if proposal_id is not None and proposal_id != proposal.proposal_id:
            raise ValueError("stale proposal id")
        if type(actual_action) is not int or not 0 <= actual_action < 18:
            raise ValueError("actual_action must be candidate integer 0..17")
        if type(actual_hold) is not int or actual_hold <= 0:
            raise ValueError("actual_hold must be measured positive logical frames")
        if type(effective_frame) is not int or effective_frame < 0:
            raise ValueError("effective_frame must be a nonnegative integer")
        if (observed is not None and effective_frame < observed) or (self.effective_frame is not None and effective_frame <= self.effective_frame):
            raise ValueError("execution frame must be causal and increasing")
        option = proposal.option if actual_option is None else actual_option
        if type(option) is not int or not 0 <= option <= 3 or type(target_adopted) is not bool:
            raise ValueError("invalid adopted option/target validity")
        self.hidden = hidden
        self.previous_candidate = actual_action
        self.previous_hold = actual_hold
        self.previous_option = option
        self.previous_target = proposal.target if target_adopted and option != 3 else (0.,0.)
        self.target_valid = target_adopted and option != 3
        self.effective_frame = effective_frame
        self._pending = None
