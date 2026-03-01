"""
Node definitions for the lead qualification pipeline.

This file is what a developer would write — 4 focused Node classes.
Everything else (verification, caching, proving, routing) is handled
by the engine automatically.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pydantic import BaseModel, Field
from aura_state import Node


class LeadData(BaseModel):
    name: str = Field(description="Full name of the lead")
    budget: int = Field(description="Total budget in USD")
    bedrooms: int = Field(description="Number of bedrooms requested")
    city: str = Field(description="Preferred city or area")
    timeline: str = Field(description="Buying timeline: immediate, 1-3 months, 3-6 months, exploring")
    pre_approved: bool = Field(description="Whether the lead has mortgage pre-approval")


class QualificationScore(BaseModel):
    budget_per_bedroom: float = Field(description="Budget divided by bedrooms")
    urgency_score: int = Field(description="1-10 urgency based on timeline")
    readiness_score: int = Field(description="1-10 readiness based on pre-approval and timeline")


class VerificationStatus(BaseModel):
    data_valid: bool = Field(description="Whether all extracted data passes validation")
    confidence: float = Field(description="Confidence score 0-1")


class RouteDecision(BaseModel):
    route: str = Field(description="hot, warm, or cold")
    reason: str = Field(description="Why this route was chosen")


class ExtractLead(Node):
    system_prompt = "Extract lead information from a sales call transcript."
    extracts = LeadData

    proof_obligations = [
        "budget > 0",
        "bedrooms >= 0",
    ]

    def handle(self, user_text, extracted_data=None, memory=None):
        data = extracted_data.model_dump() if extracted_data else {}
        return "QualifyBudget", data


class QualifyBudget(Node):
    system_prompt = "Calculate qualification metrics from lead data."
    extracts = QualificationScore
    sandbox_rule = "result = budget_per_bedroom > 0 if bedrooms > 0 else True"

    def handle(self, user_text, extracted_data=None, memory=None):
        data = memory or {}
        budget = data.get("budget", 0)
        bedrooms = data.get("bedrooms", 1) or 1
        timeline = data.get("timeline", "exploring")

        urgency = {"immediate": 10, "1-3 months": 7, "3-6 months": 4, "exploring": 1}
        readiness = 8 if data.get("pre_approved") else 3
        budget_per_bed = round(budget / bedrooms, 2) if bedrooms > 0 else float(budget)

        data["budget_per_bedroom"] = budget_per_bed
        data["urgency_score"] = urgency.get(timeline, 1)
        data["readiness_score"] = readiness
        return "VerifyData", data


class VerifyData(Node):
    system_prompt = "Verify the extracted and calculated data is consistent."

    def handle(self, user_text, extracted_data=None, memory=None):
        data = memory or {}
        data["data_valid"] = data.get("budget", 0) > 0 and data.get("name", "") != ""
        data["confidence"] = 0.95 if data["data_valid"] else 0.3
        return "RouteDecision", data


class RouteLead(Node):
    system_prompt = "Route the lead based on qualification scores."
    extracts = RouteDecision

    def handle(self, user_text, extracted_data=None, memory=None):
        data = memory or {}
        urgency = data.get("urgency_score", 1)
        readiness = data.get("readiness_score", 1)
        combined = urgency + readiness

        if combined >= 15:
            route = "hot"
            reason = f"High urgency ({urgency}) + high readiness ({readiness})"
        elif combined >= 8:
            route = "warm"
            reason = f"Moderate urgency ({urgency}) + readiness ({readiness})"
        else:
            route = "cold"
            reason = f"Low urgency ({urgency}) + low readiness ({readiness})"

        data["route"] = route
        data["reason"] = reason
        return "END", data
