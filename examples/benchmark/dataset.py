"""
Synthetic sales call transcripts with known ground-truth values.

Each entry has:
    - transcript: what the sales rep heard
    - ground_truth: the correct extracted values (for accuracy measurement)
    - expected_route: hot / warm / cold
"""

TRANSCRIPTS = [
    {
        "id": 1,
        "transcript": (
            "Hi, I'm Sarah Mitchell. We're looking for a 3-bedroom home in Austin. "
            "Our budget is around $450,000. We've been pre-approved by Chase and "
            "need to move by June because of a job relocation. We're currently "
            "renting and our lease ends May 31st."
        ),
        "ground_truth": {
            "name": "Sarah Mitchell",
            "budget": 450000,
            "bedrooms": 3,
            "city": "Austin",
            "timeline": "immediate",
            "pre_approved": True,
        },
        "expected_route": "hot",
    },
    {
        "id": 2,
        "transcript": (
            "This is James Park calling. I was curious about that listing on Oak "
            "Street. I'm not really sure what our budget is yet, maybe somewhere "
            "between 300 and 400 thousand. We have two kids so we'd need at least "
            "4 bedrooms. No rush though, probably next year sometime."
        ),
        "ground_truth": {
            "name": "James Park",
            "budget": 350000,
            "bedrooms": 4,
            "city": "unknown",
            "timeline": "exploring",
            "pre_approved": False,
        },
        "expected_route": "cold",
    },
    {
        "id": 3,
        "transcript": (
            "My name is Diana Torres. My husband and I just sold our condo for "
            "$380,000 and we're looking to upgrade to a house. Budget is $520,000, "
            "we've already talked to a lender. Looking in the Cedar Park area, "
            "want to be settled before school starts in August."
        ),
        "ground_truth": {
            "name": "Diana Torres",
            "budget": 520000,
            "bedrooms": 0,
            "city": "Cedar Park",
            "timeline": "1-3 months",
            "pre_approved": True,
        },
        "expected_route": "hot",
    },
    {
        "id": 4,
        "transcript": (
            "Hey, it's Mike Chen. I'm an investor looking at multi-family "
            "properties. Budget up to $1.2 million. I want something that "
            "cash-flows from day one. Doesn't matter where exactly as long as "
            "cap rate is above 6%. I can close in 30 days, all cash."
        ),
        "ground_truth": {
            "name": "Mike Chen",
            "budget": 1200000,
            "bedrooms": 0,
            "city": "flexible",
            "timeline": "immediate",
            "pre_approved": True,
        },
        "expected_route": "hot",
    },
    {
        "id": 5,
        "transcript": (
            "I'm Lisa Wang. Recently divorced, just starting to look at what's "
            "out there. I honestly have no idea what anything costs these days. "
            "Maybe 200k? Is that realistic for a 2-bedroom? I don't need to "
            "move right away, just gathering information."
        ),
        "ground_truth": {
            "name": "Lisa Wang",
            "budget": 200000,
            "bedrooms": 2,
            "city": "unknown",
            "timeline": "exploring",
            "pre_approved": False,
        },
        "expected_route": "cold",
    },
    {
        "id": 6,
        "transcript": (
            "Name's Robert Harris. I've been looking for about two months now. "
            "Pre-approved for $600k through Wells Fargo. Need a 3-bed in Round "
            "Rock, close to schools. Want to close before summer but not in a "
            "huge rush. Seen about 8 houses so far."
        ),
        "ground_truth": {
            "name": "Robert Harris",
            "budget": 600000,
            "bedrooms": 3,
            "city": "Round Rock",
            "timeline": "1-3 months",
            "pre_approved": True,
        },
        "expected_route": "hot",
    },
    {
        "id": 7,
        "transcript": (
            "Hi there, I'm Angela Davis. My partner and I are first-time "
            "homebuyers. We think we can afford around $350,000. We're looking "
            "at the east side of town for a starter home, 2 or 3 bedrooms. "
            "We're hoping to apply for a mortgage next month."
        ),
        "ground_truth": {
            "name": "Angela Davis",
            "budget": 350000,
            "bedrooms": 2,
            "city": "east side",
            "timeline": "1-3 months",
            "pre_approved": False,
        },
        "expected_route": "warm",
    },
    {
        "id": 8,
        "transcript": (
            "Tom Baker here. I need to relocate from Dallas to Austin by the "
            "end of next week for work. Budget is $475,000. Already "
            "pre-approved, just need something fast. 3 bedrooms minimum, "
            "prefer north Austin. My company is covering closing costs."
        ),
        "ground_truth": {
            "name": "Tom Baker",
            "budget": 475000,
            "bedrooms": 3,
            "city": "north Austin",
            "timeline": "immediate",
            "pre_approved": True,
        },
        "expected_route": "hot",
    },
    {
        "id": 9,
        "transcript": (
            "This is Karen Williams. I'm downsizing after my kids moved out. "
            "Looking for a smaller place, maybe a condo or townhome. Budget "
            "around $280,000. No mortgage needed, I'll pay cash from the sale "
            "of my current home. Sometime in the next 6 months."
        ),
        "ground_truth": {
            "name": "Karen Williams",
            "budget": 280000,
            "bedrooms": 0,
            "city": "unknown",
            "timeline": "3-6 months",
            "pre_approved": True,
        },
        "expected_route": "warm",
    },
    {
        "id": 10,
        "transcript": (
            "I'm David Nguyen. Just browsing really. My friend bought a house "
            "last year and I'm curious what's available. Probably can't buy "
            "for at least another year while I save up. Budget would be "
            "around $250k when I'm ready."
        ),
        "ground_truth": {
            "name": "David Nguyen",
            "budget": 250000,
            "bedrooms": 0,
            "city": "unknown",
            "timeline": "exploring",
            "pre_approved": False,
        },
        "expected_route": "cold",
    },
]
