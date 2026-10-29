"""
Agent-Based AI Architecture Module
Multi-Agent Coordination for Biometric Verification, Anomaly Detection,
and Autonomous Attendance Governance.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import datetime

from .config import (
    DEFAULT_WEIGHT_FACE, DEFAULT_WEIGHT_VOICE, DEFAULT_WEIGHT_LIVENESS,
    OVERALL_PASS_THRESHOLD, FACE_PASS_THRESHOLD, VOICE_PASS_THRESHOLD,
    LIVENESS_PASS_THRESHOLD
)
from .database import DatabaseManager
from .genai_engine import GenAIEngine


@dataclass
class AgentVerificationDecision:
    decision: str              # "ACCEPTED", "REJECTED", "FLAGGED"
    status: str                # "PRESENT", "LATE", "REJECTED", "FLAGGED"
    confidence_score: float
    matched_user: Optional[Dict[str, Any]]
    explanation: str
    component_scores: Dict[str, float]
    anomaly_detected: bool
    anomaly_reason: Optional[str]


class VerificationAgent:
    """
    Autonomous Biometric Decision Agent.
    Evaluates visual, acoustic, and liveness signals to adjudicate attendance punches.
    """
    def __init__(
        self,
        weight_face: float = DEFAULT_WEIGHT_FACE,
        weight_voice: float = DEFAULT_WEIGHT_VOICE,
        weight_liveness: float = DEFAULT_WEIGHT_LIVENESS,
        pass_threshold: float = OVERALL_PASS_THRESHOLD
    ):
        self.w_face = weight_face
        self.w_voice = weight_voice
        self.w_liveness = weight_liveness
        self.pass_threshold = pass_threshold

    def evaluate(
        self,
        face_user: Optional[Dict[str, Any]],
        face_score: float,
        voice_user: Optional[Dict[str, Any]],
        voice_score: float,
        liveness_score: float,
        is_live: bool,
        mode: str = "MULTIMODAL"  # "MULTIMODAL", "FACE_ONLY", "VOICE_ONLY"
    ) -> AgentVerificationDecision:
        """
        Synthesize multi-sensor inputs into a verified decision.
        """
        component_scores = {
            "face_score": round(face_score, 3),
            "voice_score": round(voice_score, 3),
            "liveness_score": round(liveness_score, 3)
        }

        # Step 1: Liveness Safety Gate (Anti-Spoofing barrier)
        if mode in ["MULTIMODAL", "FACE_ONLY"] and not is_live:
            return AgentVerificationDecision(
                decision="REJECTED",
                status="REJECTED",
                confidence_score=round(liveness_score, 3),
                matched_user=face_user,
                explanation="Security Alert: Liveness verification failed. Potential 2D photo presentation or static replay attack detected.",
                component_scores=component_scores,
                anomaly_detected=True,
                anomaly_reason="SPOOF_ATTEMPT"
            )

        # Step 2: Identity Consistency Validation
        if mode == "MULTIMODAL" and face_user and voice_user:
            if face_user.get("user_id") != voice_user.get("user_id"):
                # Discrepancy: Face belongs to User A, but voice belongs to User B!
                return AgentVerificationDecision(
                    decision="FLAGGED",
                    status="FLAGGED",
                    confidence_score=round((face_score + voice_score) / 2.0, 3),
                    matched_user=face_user,
                    explanation=f"Identity Conflict: Visual signature matched '{face_user.get('name')}' ({face_score:.0%}), but voice acoustic matched '{voice_user.get('name')}' ({voice_score:.0%}). Possible proxy punch attempt.",
                    component_scores=component_scores,
                    anomaly_detected=True,
                    anomaly_reason="IDENTITY_MISMATCH"
                )

        # Step 3: Compute Combined Confidence
        if mode == "MULTIMODAL":
            combined_score = (
                self.w_face * face_score +
                self.w_voice * voice_score +
                self.w_liveness * liveness_score
            )
            matched_user = face_user or voice_user
        elif mode == "FACE_ONLY":
            norm_w = self.w_face + self.w_liveness
            combined_score = (self.w_face * face_score + self.w_liveness * liveness_score) / (norm_w if norm_w > 0 else 1.0)
            matched_user = face_user
        else:  # VOICE_ONLY
            combined_score = voice_score
            matched_user = voice_user

        combined_score = round(max(0.0, min(1.0, combined_score)), 3)
        component_scores["total_score"] = combined_score

        # Step 4: Adjudicate Decision
        if matched_user is None or combined_score < self.pass_threshold:
            return AgentVerificationDecision(
                decision="REJECTED",
                status="REJECTED",
                confidence_score=combined_score,
                matched_user=matched_user,
                explanation=f"Verification Failed: Combined biometric score ({combined_score:.1%}) is below authorization threshold ({self.pass_threshold:.1%}).",
                component_scores=component_scores,
                anomaly_detected=False,
                anomaly_reason=None
            )

        explanation = (
            f"Verification Successful for {matched_user.get('name')} ({matched_user.get('user_id')}). "
            f"Confidence: {combined_score:.1%} [Face: {face_score:.0%}, Voice: {voice_score:.0%}, Liveness: {liveness_score:.0%}]."
        )

        return AgentVerificationDecision(
            decision="ACCEPTED",
            status="PRESENT",
            confidence_score=combined_score,
            matched_user=matched_user,
            explanation=explanation,
            component_scores=component_scores,
            anomaly_detected=False,
            anomaly_reason=None
        )


class AnomalyDetectionAgent:
    """
    Fraud, Policy & Security Inspector Agent.
    Monitors duplicate punches, proxy attempts, and operational anomalies.
    """
    def __init__(self, db: DatabaseManager):
        self.db = db

    def inspect_punch(self, user_id: str, decision: AgentVerificationDecision) -> Tuple[bool, str]:
        """
        Audit the verification decision before recording.
        """
        if decision.anomaly_detected and decision.anomaly_reason:
            self.db.record_anomaly(
                user_id=user_id,
                anomaly_type=decision.anomaly_reason,
                severity="HIGH" if decision.anomaly_reason == "SPOOF_ATTEMPT" else "MEDIUM",
                details=decision.explanation
            )
            return False, decision.explanation

        # Check repeat punch cooldown policy
        can_punch, cooldown_msg = self.db.can_record_punch(user_id)
        if not can_punch:
            self.db.record_anomaly(
                user_id=user_id,
                anomaly_type="RAPID_DUPLICATE",
                severity="LOW",
                details=cooldown_msg
            )
            return False, cooldown_msg

        return True, "Passed Security Inspection"


class MultimodalAgentCoordinator:
    """
    Central Coordinator managing perception engines, decision agents, and persistent memory.
    """
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self.verification_agent = VerificationAgent()
        self.anomaly_agent = AnomalyDetectionAgent(self.db)
        self.genai_engine = GenAIEngine()

    def process_punch(
        self,
        face_user: Optional[Dict[str, Any]],
        face_score: float,
        voice_user: Optional[Dict[str, Any]],
        voice_score: float,
        liveness_score: float,
        is_live: bool,
        mode: str = "MULTIMODAL"
    ) -> AgentVerificationDecision:
        # 1. Run Verification Agent
        decision = self.verification_agent.evaluate(
            face_user=face_user,
            face_score=face_score,
            voice_user=voice_user,
            voice_score=voice_score,
            liveness_score=liveness_score,
            is_live=is_live,
            mode=mode
        )

        user_id = decision.matched_user.get("user_id") if decision.matched_user else "UNKNOWN"

        # 2. Run Anomaly Agent
        if decision.decision == "ACCEPTED":
            allowed, audit_msg = self.anomaly_agent.inspect_punch(user_id, decision)
            if not allowed:
                decision.decision = "FLAGGED"
                decision.status = "FLAGGED"
                decision.explanation += f" [Policy Flag: {audit_msg}]"
                decision.anomaly_detected = True

        # 3. Persist to Database if user is identified
        if user_id != "UNKNOWN":
            self.db.record_attendance(
                user_id=user_id,
                face_score=decision.component_scores.get("face_score", 0.0),
                voice_score=decision.component_scores.get("voice_score", 0.0),
                liveness_score=decision.component_scores.get("liveness_score", 0.0),
                total_score=decision.confidence_score,
                verification_mode=mode,
                status=decision.status,
                notes=decision.explanation,
                anomaly_flag=1 if decision.anomaly_detected else 0
            )

        return decision
