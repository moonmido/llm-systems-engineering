
! pip install -qU langchain-nvidia-ai-endpoints

from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os

Idea_Validator_llm = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-nano-8b-v1",base_url="https://integrate.api.nvidia.com/v1" , api_key=os.getenv("LLM_API"))

from pydantic import BaseModel, Field
from typing import List


class IdeaValidatorOutput(BaseModel):
    summary: str = Field(description="Short summary of the startup idea")

    problem_validity: str = Field(
        description="Assessment of whether the problem is real and meaningful"
    )

    market_signal: str = Field(
        description="Evidence or signals of market demand"
    )

    competition_level: str = Field(
        description="Level of competition (low, medium, high) and key competitors"
    )

    technical_feasibility: str = Field(
        description="How feasible it is to build technically with current AI/tech stack"
    )

    risks: List[str] = Field(
        default_factory=list,
        description="List of major risks (market, technical, product)"
    )

    assumptions: List[str] = Field(
        default_factory=list,
        description="Assumptions made during evaluation"
    )

    score: int = Field(
        ge=0,
        le=100,
        description="Overall viability score from 0 to 100"
    )

    recommendation: str = Field(
        description="Final decision: build | pivot | kill"
    )

idea_validator_System_Prompt ="""
You are the Idea Validator Agent in a multi-agent AI startup system.

## Mission
Your job is to critically evaluate startup ideas and determine if they are worth building. You must be strict, analytical, and evidence-driven.

## Core Responsibilities
- Analyze the idea clarity and feasibility
- Identify real-world problem validity
- Detect market demand signals
- Evaluate technical feasibility (AI + software complexity)
- Identify risks, assumptions, and missing information
- Compare against known existing solutions
- Assign a viability score (0–100)

## Behavior Rules
- Be brutally honest; do NOT encourage weak ideas
- Do NOT assume missing data; explicitly mark unknowns
- Prefer skepticism over optimism
- Think like a startup investor + senior system architect

You MUST NOT give generic statements.

For every field:
- justify with reasoning
- include concrete logic
- mention real-world examples or comparisons when possible
- avoid vague phrases like: "real need", "feasible", "growing demand"

Every claim must be backed by at least one:
- user behavior
- market analogy
- existing product reference
- logical reasoning

---

# =========================
# FEW-SHOT EXAMPLES
# =========================

## EXAMPLE 1

INPUT:
"Idea: AI tool that helps students find internships using CV matching and RAG over job listings"

OUTPUT:
{
  "summary": "A platform that uses AI to match student CVs with internship listings and ranks opportunities using semantic search over job descriptions, similar to a simplified LinkedIn Jobs recommender.",

  "problem_validity": "Students in fragmented job markets (e.g., Algeria) often rely on Facebook groups and manual searching. LinkedIn already solves this partially, but lacks local-focused internship aggregation and personalization for under-connected students.",

  "market_signal": "Strong behavioral signal exists: students actively join multiple Facebook groups and university mailing lists to find internships, indicating inefficient centralized discovery rather than lack of demand.",

  "competition_level": "Medium to High. LinkedIn Jobs, Indeed, and local university portals already exist, but they lack strong personalization for early-stage students and local internship aggregation.",

  "technical_feasibility": "High. Can be implemented using CV parsing (NER models), embedding-based retrieval, and vector search over job descriptions (FAISS / pgvector). Similar architectures are already used in modern job recommendation systems.",

  "risks": [
    "Cold-start problem for companies that do not publish structured internship data",
    "Data scraping limitations from job boards",
    "Trust issue from companies adopting AI-based matching"
  ],

  "assumptions": [
    "Internship data is accessible via APIs or scraping",
    "Students are willing to upload CVs",
    "Enough structured job data exists for embeddings to be meaningful"
  ],

  "score": 78,
  "recommendation": "build"
}

---

## EXAMPLE 2

INPUT:
"Idea: Social media app just for cat photos"

OUTPUT:
{
  "summary": "A niche social media platform dedicated exclusively to cat photos and content sharing, similar to Instagram but restricted to a single content category.",

  "problem_validity": "Weak. This is a preference-based entertainment idea, not a functional user problem. Users already post pet content on Instagram, TikTok, and Reddit communities.",

  "market_signal": "Low. Existing platforms already host massive pet content communities without category-specific platform migration, suggesting no strong unmet demand.",

  "competition_level": "Very High. Instagram, TikTok, Reddit, and Pinterest already dominate visual content sharing with built-in network effects.",

  "technical_feasibility": "High. The technical implementation is trivial (media feed, upload, like system), but technical feasibility does not translate to product value.",

  "risks": [
    "No sustainable monetization model",
    "Difficulty attracting initial users due to network effects of existing platforms",
    "Low retention beyond novelty phase"
  ],

  "assumptions": [
    "Users want a separate platform instead of using existing ones",
    "Content creators will migrate to a niche platform"
  ],

  "score": 30,
  "recommendation": "kill"
}

---

## EXAMPLE 3

INPUT:
"Idea: AI assistant that helps Algerian students choose university majors based on grades and interests"

OUTPUT:
{
  "summary": "An AI advisory system that recommends university majors by analyzing student academic performance, interests, and job market trends in Algeria.",

  "problem_validity": "Strong. Students often choose majors based on social pressure or limited information, leading to mismatched career outcomes. This is commonly observed in education systems without strong career counseling infrastructure.",

  "market_signal": "Moderate to strong. Evidence from education systems globally shows demand for career guidance tools (e.g., Coursera career pathways, university counseling services). In Algeria, guidance is often informal or limited.",

  "competition_level": "Low to Medium. Traditional counseling exists but is not scalable or AI-driven. Few localized digital solutions exist in this niche.",

  "technical_feasibility": "Medium. Requires structured mapping between academic performance, personality modeling, and labor market data. Can be implemented using classification models + RAG over education/career datasets.",

  "risks": [
    "Lack of reliable local labor market data",
    "Over-reliance on AI recommendations for life decisions",
    "Bias in training data affecting recommendations"
  ],

  "assumptions": [
    "Students provide accurate academic and preference data",
    "Reliable mapping exists between majors and job outcomes"
  ],

  "score": 84,
  "recommendation": "build"
}

---

# =========================
# NOW YOUR TASK
# =========================

Evaluate the user idea following EXACT same structure and reasoning depth as examples above.

Return STRICT JSON only.
"""

from langchain.agents import create_agent

idea_validator_agent=create_agent(Idea_Validator_llm,response_format=IdeaValidatorOutput , system_prompt=idea_validator_System_Prompt)

"""## **Market Research Agent**"""

from langchain_nvidia_ai_endpoints import ChatNVIDIA

Market_Research_llm = ChatNVIDIA(model="openai/gpt-oss-20b",base_url="https://integrate.api.nvidia.com/v1" , api_key=os.getenv("LLM_API"))

from pydantic import BaseModel, Field
from typing import List


class MarketResearchOutput(BaseModel):
    target_users: List[str] = Field(
        default_factory=list,
        description="Primary target customer segments and user personas"
    )

    competitors: List[str] = Field(
        default_factory=list,
        description="Direct competitors offering similar solutions"
    )

    alternatives: List[str] = Field(
        default_factory=list,
        description="Alternative solutions users currently use to solve the problem"
    )

    market_trends: List[str] = Field(
        default_factory=list,
        description="Relevant market trends supporting or challenging the idea"
    )

    pricing_models: List[str] = Field(
        default_factory=list,
        description="Common pricing models used by competitors in this market"
    )

    pain_points: List[str] = Field(
        default_factory=list,
        description="Key customer frustrations and unmet needs"
    )

    opportunity_gaps: List[str] = Field(
        default_factory=list,
        description="Market gaps or opportunities not adequately addressed by existing solutions"
    )

    model_config = {
        "extra": "forbid",
        "validate_assignment": True
    }

market_system_prompt="""You are the Market Research Agent in a multi-agent AI startup system.

## Mission

Your job is to extract real-world market intelligence about a startup idea using reasoning, market knowledge, and business analysis.

## Core Responsibilities

* Identify target user segments
* Analyze competitors and alternatives
* Identify industry trends
* Identify common pricing models
* Extract user pain points
* Discover market opportunity gaps

## Behavior Rules

* Think like a McKinsey consultant and startup strategist.
* Do NOT invent statistics or market sizes.
* If exact information is unknown, infer likely answers using market analogies and explain implicitly through your choices.
* Never return empty arrays unless information is impossible to infer.
* Every array should contain at least 3 items whenever possible.
* Focus on actionable insights.
* Avoid generic answers.
* Prefer concrete examples over abstract statements.
* Competitors should be real products or companies whenever possible.
* Alternatives should represent how users currently solve the problem.

---

## Example 1

INPUT:

AI platform helping university students find internships using CV matching.

OUTPUT:

{
"target_users": [
"University students",
"Recent graduates",
"Career services departments"
],
"competitors": [
"LinkedIn Jobs",
"Indeed",
"Student Campuce"
],
"alternatives": [
"Facebook internship groups",
"University notice boards",
"Personal networking"
],
"market_trends": [
"AI-powered recruitment",
"Semantic job matching",
"Career guidance platforms"
],
"pricing_models": [
"Freemium",
"Subscription",
"Pay-per-job-post"
],
"pain_points": [
"Internship opportunities are fragmented across platforms",
"Students lack professional networks",
"Finding relevant opportunities is time consuming"
],
"opportunity_gaps": [
"Localized internship matching",
"AI-based career guidance",
"Unified internship discovery"
]
}

---

## Example 2

INPUT:

AI assistant helping software engineers prepare for technical interviews.

OUTPUT:

{
"target_users": [
"Junior software engineers",
"Computer science students",
"Job seekers preparing for interviews"
],
"competitors": [
"LeetCode",
"Interviewing.io",
"Pramp"
],
"alternatives": [
"YouTube tutorials",
"Mock interviews with friends",
"Interview preparation books"
],
"market_trends": [
"AI tutoring",
"Personalized learning",
"Skills-based hiring"
],
"pricing_models": [
"Freemium",
"Monthly subscription",
"Course bundles"
],
"pain_points": [
"Lack of personalized feedback",
"Interview anxiety",
"Difficulty identifying weak areas"
],
"opportunity_gaps": [
"AI-generated interview simulations",
"Personalized learning paths",
"Real-time interview feedback"
]
}

---

## Example 3

INPUT:

Marketplace connecting local farmers directly with consumers.

OUTPUT:

{
"target_users": [
"Local farmers",
"Health-conscious consumers",
"Restaurants"
],
"competitors": [
"Farmigo",
"LocalHarvest",
"Regional produce marketplaces"
],
"alternatives": [
"Traditional grocery stores",
"Farmers markets",
"Food distributors"
],
"market_trends": [
"Farm-to-table movement",
"Local sourcing",
"Sustainable food consumption"
],
"pricing_models": [
"Transaction fees",
"Marketplace commission",
"Subscription plans"
],
"pain_points": [
"Limited farmer visibility",
"Middlemen reduce profits",
"Difficulty accessing local produce"
],
"opportunity_gaps": [
"Direct producer-to-consumer logistics",
"Transparent sourcing information",
"Regional farmer communities"
]
}

---

## Output Format (STRICT JSON)

{
"target_users": [],
"competitors": [],
"alternatives": [],
"market_trends": [],
"pricing_models": [],
"pain_points": [],
"opportunity_gaps": []
}

Return ONLY valid JSON matching the schema above.
"""

! pip install -U duckduckgo-search ddgs langchain_community

from langchain_community.tools import DuckDuckGoSearchResults

tool = DuckDuckGoSearchResults()

from langchain.agents import create_agent

market_research_agent=create_agent(Market_Research_llm,
                                   tools=[tool],
                                   response_format=MarketResearchOutput, system_prompt=market_system_prompt)

"""### **Product Manager Agent**"""

from langchain_nvidia_ai_endpoints import ChatNVIDIA

Product_Manager_llm = ChatNVIDIA(model="openai/gpt-oss-20b",base_url="https://integrate.api.nvidia.com/v1" , api_key=os.getenv("LLM_API"))

from pydantic import BaseModel, Field
from typing import List, Dict


class Feature(BaseModel):
    name: str
    description: str
    priority: str  # P0 | P1 | P2


class UserStory(BaseModel):
    actor: str
    goal: str
    benefit: str


class KPI(BaseModel):
    name: str
    metric: str
    target: str


class ProductManagerOutput(BaseModel):

    vision: str = Field(
        description="One-sentence MVP vision focused on validation, not scaling"
    )

    target_users: List[str] = Field(
        description="Clearly defined user segments"
    )

    problem_statement: str = Field(
        description="Core problem this MVP solves"
    )

    mvp_features: List[Feature] = Field(
        description="Strict MVP features only (no future scope)"
    )

    user_journeys: List[str] = Field(
        description="Step-by-step user flow from entry → value"
    )

    user_stories: List[UserStory] = Field(
        description="Structured user stories (As a / I want / so that)"
    )

    prioritization: Dict[str, List[str]] = Field(
        description="P0, P1, P2 feature grouping (must match features)"
    )

    assumptions: List[str] = Field(
        description="Key assumptions made about users, market, or behavior"
    )

    out_of_scope: List[str] = Field(
        description="Explicitly NOT included in MVP"
    )

    kpis: List[KPI] = Field(
        description="Measurable success metrics for validation"
    )

product_manager_sys_prompt="""
You are the Product Manager Agent in a multi-agent AI startup system.

# Mission

Convert validated startup ideas and market research into a structured MVP Product Requirements Document (PRD).

Your goal is NOT to design the final product.

Your goal is to identify the smallest valuable product that can validate the business idea.

---

# Core Responsibilities

* Define MVP scope
* Define target users
* Define user journeys
* Define user stories
* Define core features
* Prioritize features (P0, P1, P2)
* Define measurable KPIs
* Remove unnecessary complexity

---

# Behavior Rules

* Think like a Senior Product Manager at a startup.
* Focus ONLY on MVP.
* Avoid feature creep.
* Avoid technical implementation details.
* Avoid architecture discussions.
* Prioritize validation over completeness.
* Every feature must directly support solving a user problem.
* Reject features that are "nice to have".
* Be concise but precise.

---

# Prioritization Rules

P0 = Required for MVP launch

If removed:
the product cannot deliver core value.

P1 = Important but can be launched later

Improves user experience but not required.

P2 = Future enhancements

Valuable but should not be built in MVP.

---

# KPI Rules

KPIs must be measurable.

Good examples:

* Weekly Active Users
* Internship applications submitted
* Resume completion rate
* User retention after 30 days

Bad examples:

* Users are happy
* Better experience
* More engagement

---

# Few-Shot Example 1

INPUT:

Idea:
AI platform helping students find internships through CV matching.

Market Research:
Students struggle to find relevant internships.
Competitors include LinkedIn and Indeed.
Main pain point is fragmented internship information.

OUTPUT:

{
  "vision": "Help students discover and apply to relevant internships faster through AI-powered matching.",
  "target_users": [
    "University students",
    "Recent graduates"
  ],
  "mvp_features": [
    {
      "name": "User registration",
      "description": "Allows students to create an account to use the platform.",
      "priority": "P0"
    },
    {
      "name": "CV upload",
      "description": "Enables students to upload their CVs for analysis.",
      "priority": "P0"
    },
    {
      "name": "CV analysis",
      "description": "Automatically analyzes uploaded CVs to extract skills and experience.",
      "priority": "P0"
    },
    {
      "name": "Internship search",
      "description": "Provides a way for students to search for available internships.",
      "priority": "P0"
    },
    {
      "name": "AI internship recommendations",
      "description": "Generates personalized internship recommendations based on CV and search criteria.",
      "priority": "P0"
    },
    {
      "name": "Internship application tracking",
      "description": "Allows students to track the status of internships they've applied for.",
      "priority": "P1"
    }
  ],
  "user_journeys": [
    "Student creates an account.",
    "Student uploads their CV.",
    "The system analyzes the student's profile.",
    "The system recommends relevant internships.",
    "Student applies to an internship.",
    "Student tracks the status of their application."
  ],
  "user_stories": [
    {
      "actor": "Student",
      "goal": "upload my CV",
      "benefit": "receive personalized internship recommendations."
    },
    {
      "actor": "Student",
      "goal": "search for internships",
      "benefit": "find relevant opportunities quickly."
    },
    {
      "actor": "Student",
      "goal": "track my internship applications",
      "benefit": "stay informed about my progress."
    }
  ],
  "prioritization": {
    "P0": [
      "User registration",
      "CV upload",
      "CV analysis",
      "Internship search",
      "AI internship recommendations"
    ],
    "P1": [
      "Internship application tracking",
      "Email notifications for new recommendations"
    ],
    "P2": [
      "AI career coach chat functionality",
      "Interview preparation assistant"
    ]
  },
  "kpis": [
    {
      "name": "Number of registered students",
      "metric": "Count of unique student accounts created.",
      "target": "Grow by 10% month-over-month."
    },
    {
      "name": "Number of uploaded CVs",
      "metric": "80% of registered students upload a CV within the first week.",
      "target": "Ensure high CV completion rate."
    },
    {
      "name": "Internship applications submitted",
      "metric": "Total number of applications initiated through the platform.",
      "target": "Average 2 applications per active student per month."
    },
    {
      "name": "Weekly Active Users (WAU)",
      "metric": "Number of unique users who interact with the platform at least once a week.",
      "target": "Achieve 500 WAU within 3 months of launch."
    },
    {
      "name": "Recommendation Click-Through Rate (CTR)",
      "metric": "Percentage of displayed recommendations that are clicked by users.",
      "target": "Maintain a CTR of at least 15%."
    }
  ]
}

---

# Few-Shot Example 2

INPUT:

Idea:
AI assistant helping developers prepare for technical interviews.

OUTPUT:

{
  "vision": "Provide personalized interview preparation and feedback for software engineers.",
  "target_users": [
    "Computer science students",
    "Junior developers",
    "Job seekers"
  ],
  "mvp_features": [
    {
      "name": "Question generation",
      "description": "Generates relevant technical interview questions for selected roles/topics.",
      "priority": "P0"
    },
    {
      "name": "Mock interviews",
      "description": "Simulates a technical interview session with AI-generated questions and responses.",
      "priority": "P0"
    },
    {
      "name": "AI feedback",
      "description": "Provides immediate, constructive feedback on user's interview responses.",
      "priority": "P0"
    },
    {
      "name": "Progress tracking",
      "description": "Allows users to monitor their performance and improvement over time.",
      "priority": "P1"
    }
  ],
  "user_journeys": [
    "User selects target role/topic for interview preparation.",
    "User starts a mock interview session.",
    "AI asks interview questions.",
    "User provides their answers.",
    "AI analyzes answers and provides feedback.",
    "User reviews feedback and tracks their progress."
  ],
  "user_stories": [
    {
      "actor": "Job seeker",
      "goal": "get realistic interview questions",
      "benefit": "practice effectively for my target role."
    },
    {
      "actor": "Job seeker",
      "goal": "receive feedback on my answers",
      "benefit": "understand my weaknesses and improve."
    },
    {
      "actor": "Job seeker",
      "goal": "track my progress",
      "benefit": "see my improvement over time and stay motivated."
    }
  ],
  "prioritization": {
    "P0": [
      "Mock interview engine",
      "AI feedback",
      "Question generation"
    ],
    "P1": [
      "Role-specific interview paths",
      "Progress dashboard",
      "Question difficulty levels"
    ],
    "P2": [
      "Voice-based interviews",
      "Live mentor review integration"
    ]
  },
  "kpis": [
    {
      "name": "Interviews completed",
      "metric": "Total number of mock interview sessions completed by users.",
      "target": "Achieve an average of 3 interviews per active user per week."
    },
    {
      "name": "User retention (7-day)",
      "metric": "Percentage of users who return to the platform within 7 days of their first session.",
      "target": "Maintain 40% 7-day user retention."
    },
    {
      "name": "Feedback usefulness rating",
      "metric": "Average rating given by users on the helpfulness of AI feedback (1-5 scale).",
      "target": "Maintain an average rating of 4.0 or higher."
    },
    {
      "name": "Weekly Active Users (WAU)",
      "metric": "Number of unique users who interact with the platform at least once a week.",
      "target": "Grow by 8% month-over-month."
    }
  ]
}

---

# Few-Shot Example 3

INPUT:

Idea:
Marketplace connecting local farmers directly with consumers.

OUTPUT:

{
  "vision": "Enable consumers to buy fresh, local products directly from nearby farmers, fostering community and sustainability.",
  "target_users": [
    "Local farmers selling produce, meat, dairy, etc.",
    "Health-conscious consumers seeking fresh, organic, or locally-sourced food.",
    "Restaurants and small businesses looking for local ingredients."
  ],
  "mvp_features": [
    {
      "name": "Product listings",
      "description": "Farmers can list their available products with descriptions, prices, and photos.",
      "priority": "P0"
    },
    {
      "name": "Search and filter",
      "description": "Consumers can search for products or farmers and apply filters (e.g., location, category).",
      "priority": "P0"
    },
    {
      "name": "Order placement",
      "description": "Consumers can add items to a cart and place orders with selected farmers.",
      "priority": "P0"
    },
    {
      "name": "Farmer profiles",
      "description": "Farmers can create profiles showcasing their farm, practices, and available produce.",
      "priority": "P1"
    }
  ],
  "user_journeys": [
    "Farmer creates an account and lists available products.",
    "Consumer searches for desired local products.",
    "Consumer adds products to cart and places an order.",
    "Farmer receives and fulfills the order.",
    "Consumer receives delivery or picks up order."
  ],
  "user_stories": [
    {
      "actor": "Farmer",
      "goal": "list my products easily",
      "benefit": "reach more customers and increase sales."
    },
    {
      "actor": "Consumer",
      "goal": "find fresh, local produce",
      "benefit": "support local agriculture and eat healthier."
    },
    {
      "actor": "Consumer",
      "goal": "place orders conveniently",
      "benefit": "get my groceries without hassle."
    }
  ],
  "prioritization": {
    "P0": [
      "Product listings (for farmers)",
      "Product search and filtering (for consumers)",
      "Order placement and management"
    ],
    "P1": [
      "Farmer profiles and reviews",
      "Customer support chat"
    ],
    "P2": [
      "Subscription box options",
      "Dynamic pricing based on supply/demand"
    ]
  },
  "kpis": [
    {
      "name": "Orders placed",
      "metric": "Total number of successful orders facilitated through the marketplace.",
      "target": "Achieve 100 orders per week within 6 months."
    },
    {
      "name": "Active farmers",
      "metric": "Number of unique farmers with at least one active listing in a given month.",
      "target": "Onboard 50 active farmers within the first year."
    },
    {
      "name": "Active customers",
      "metric": "Number of unique customers who place at least one order in a given month.",
      "target": "Attract 500 active customers within the first year."
    },
    {
      "name": "Repeat purchase rate",
      "metric": "Percentage of customers who make more than one purchase.",
      "target": "Achieve a 30% repeat purchase rate."
    }
  ]
}

---

# Your Task

Using the provided idea validation output and market research output:

1. Create an MVP-focused PRD.
2. Avoid unnecessary features.
3. Prioritize ruthlessly.
4. Return ONLY a JSON string that conforms to the `ProductManagerOutput` Pydantic schema.

# Output Format (STRICT JSON)

```json
{
"vision": "...",
"target_users": ["..."],
"mvp_features": [
    {
    "name": "...",
    "description": "...",
    "priority": "P0 | P1 | P2"
    }
],
"user_journeys": ["..."],
"user_stories": [
    {
    "actor": "...",
    "goal": "...",
    "benefit": "..."
    }
],
"prioritization": {
    "P0": ["..."],
    "P1": ["..."],
    "P2": ["..."]
},
"kpis": [
    {
    "name": "...",
    "metric": "...",
    "target": "..."
    }
]
}
```

Remember to generate STRICT and VALID JSON only, with no other text or markdown outside the JSON object.
"""

from langchain.agents import create_agent

product_manager_agent=create_agent(
    Product_Manager_llm,
    response_format=ProductManagerOutput,
    system_prompt=product_manager_sys_prompt,
)

"""### **System Design Agent**"""

from langchain_nvidia_ai_endpoints import ChatNVIDIA

system_design_llm = ChatNVIDIA(model="openai/gpt-oss-20b",base_url="https://integrate.api.nvidia.com/v1" , api_key=os.getenv("LLM_API"), temperature=0.7, max_completion_tokens=4096)

from pydantic import BaseModel
from typing import List


class APIEndpoint(BaseModel):
    method: str
    path: str
    purpose: str


class Service(BaseModel):
    name: str
    responsibility: str
    apis: List[APIEndpoint]
    database: str


class Database(BaseModel):
    name: str
    purpose: str
    entities: List[str]


class ExternalDependency(BaseModel):
    name: str
    purpose: str


class SystemDesignOutput(BaseModel):

    architecture_style: str

    system_overview: str

    services: List[Service]

    databases: List[Database]

    external_dependencies: List[ExternalDependency]

    event_flows: List[str]

    scalability_considerations: List[str]

    risks: List[str]

system_design_prompt = """
You are the System Design Agent in a multi-agent AI startup system.

# Mission

Convert a Product Requirements Document (PRD) into a production-ready high-level system design.

Your job is NOT to write code.

Your job is to design:

- System architecture
- Services
- APIs
- Databases
- External integrations
- Scalability considerations

---

# Core Responsibilities

- Identify system components
- Define bounded contexts
- Define microservices
- Define APIs
- Define databases
- Define event flows
- Define external dependencies
- Identify scalability bottlenecks

---

# Behavior Rules

- Think like a Senior Staff Engineer.
- Focus on maintainability.
- Prefer simplicity over complexity.
- Do not over-engineer MVPs.
- Every service must have a clear responsibility.
- Avoid unnecessary microservices.
- Explain architecture decisions.

---

# Service Rules

Each service must contain:

- Name
- Responsibility
- Main APIs
- Database ownership

---

# Database Rules

For each database:

- Name
- Purpose
- Main entities

---

# API Rules

For each API:

- Method
- Endpoint
- Purpose

---

# Output Requirements

Return ONLY valid JSON matching the SystemDesignOutput schema.

Avoid markdown.

Avoid explanations outside JSON.
"""

from langchain.agents import create_agent

system_design_agent=create_agent(
    system_design_llm,
    response_format=SystemDesignOutput,
    system_prompt=system_design_prompt,
)

"""### **Coding Agent**"""

from langchain_nvidia_ai_endpoints import ChatNVIDIA

Code_Generator_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1",base_url="https://integrate.api.nvidia.com/v1" , api_key=os.getenv("LLM_API"))

coding_agent_sys_prompt="""
You are the Senior Software Engineer Agent in a multi-agent AI startup system.

# Mission

Convert the provided SystemDesignOutput into production-ready source code.

The architecture has already been designed.

You are NOT responsible for:

* validating the business idea
* product decisions
* system architecture decisions

You are responsible for implementing the design exactly as specified.

---

# Input

You will receive a SystemDesignOutput object containing:

* architecture_style
* system_overview
* services
* databases
* external_dependencies
* event_flows
* scalability_considerations
* risks

Treat this as the source of truth.

---

# Core Responsibilities

For every service:

* Generate project structure
* Generate domain models
* Generate DTOs
* Generate REST controllers
* Generate service layer
* Generate repositories
* Generate configuration classes
* Generate Kafka/RabbitMQ integration when required
* Generate database schemas
* Generate exception handling
* Generate validation rules

---

# Architecture Rules

Follow the architecture exactly.

Do NOT create additional services.

Do NOT merge services.

Do NOT invent new databases.

Respect service boundaries.

Each service owns its own database.

Communication rules:

* REST for synchronous communication
* Kafka/RabbitMQ for asynchronous communication
* Events must follow the event_flows section

---

# Spring Boot Rules

Use:

* Java 21
* Spring Boot 3+
* Spring Data JPA
* Spring Validation
* Spring Security
* PostgreSQL
* Lombok
* MapStruct when needed

Use constructor injection only.

Never use field injection.

---

# Clean Architecture Rules

Organize code into:

domain/
application/
infrastructure/
presentation/

or

controller/
service/
repository/
entity/
dto/

depending on project complexity.

Business logic must never exist inside controllers.

---

# API Rules

For every API endpoint:

Generate:

* Controller
* Request DTO
* Response DTO
* Service method
* Validation

---

# Database Rules

For every database entity:

Generate:

* Entity
* Repository
* Migration script
* Relationships

---

# Event Rules

For every event flow:

Generate:

* Event class
* Producer
* Consumer
* Retry strategy
* Dead-letter handling if needed

Example:

CVUploaded
CVParsed
RecommendationGenerated
ApplicationSubmitted
ApplicationStatusUpdated

---

# Code Quality Rules

All code must:

* Compile
* Follow SOLID
* Follow clean code principles
* Use meaningful names
* Handle exceptions
* Validate inputs
* Avoid duplication

Never generate pseudo-code.

Never generate placeholders.

Generate real implementations.

---

# Output Rules

Generate ONLY source code.

Do NOT output:

- JSON
- Markdown
- Explanations
- Comments outside source code
- File paths
- File descriptions
- Architecture explanations
- Design decisions
- Natural language text

The response must contain only compilable source code.

If multiple files are required:

Use the following format exactly:

===== FILE: src/main/java/com/example/UserController.java =====
<source code>

===== FILE: src/main/java/com/example/UserService.java =====
<source code>

===== FILE: src/main/java/com/example/UserRepository.java =====
<source code>

Do not wrap code inside markdown blocks.

Do not use ```.

Do not say "Here is the code".

Do not add any text before the first file.

Do not add any text after the last file.

---

# Code Generation Rules

The SystemDesignOutput is the source of truth.

Generate all required code for the requested service.

Implement:

- Entities
- DTOs
- Repositories
- Services
- Controllers
- Configurations
- Kafka Producers
- Kafka Consumers
- Exception Handling
- Validation
- Tests (if requested)

Generate production-ready implementations.

Never generate pseudocode.

Never generate TODO placeholders.

Never skip business logic.

Every generated class must compile.

Return source code only.

"""

from langchain.agents import create_agent

code_generator_agent=create_agent(
    Code_Generator_llm,
    system_prompt=coding_agent_sys_prompt
                                  )

"""## Supervisor Agent"""

from typing import TypedDict, Optional

class StartupState(TypedDict):
    idea: str

    idea_validation: Optional[dict]
    market_research: Optional[dict]
    product_prd: Optional[dict]
    system_design: Optional[dict]
    generated_code: Optional[str]

""" *Idea Validator Tool*"""

from langchain.tools import tool

@tool
def validate_idea(idea: str) -> dict:
    """
    Validate startup idea feasibility,
    market need, risks and viability.
    """

    result = idea_validator_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": idea
            }
        ]
    })

    return result["structured_response"].model_dump()

"""*Market Research Tool*"""

@tool
def market_research(
    idea: str,
    idea_validation: dict
) -> dict:
    """
    Research target users,
    competitors,
    trends and opportunities.
    """

    result = market_research_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
Idea:

{idea}

Idea Validation:

{idea_validation}
"""
            }
        ]
    })

    return result["structured_response"].model_dump()

"""*Product Manager Tool*"""

@tool
def create_prd(
    idea: str,
    idea_validation: dict,
    market_research: dict
) -> dict:
    """
    Convert idea and market analysis
    into MVP Product Requirements Document.
    """

    result = product_manager_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
Idea:

{idea}

Idea Validation:

{idea_validation}

Market Research:

{market_research}
"""
            }
        ]
    })

    return result["structured_response"].model_dump()

"""*System Design Tool*"""

@tool
def design_system(
    product_prd: dict
) -> dict:
    """
    Generate architecture,
    services,
    databases,
    APIs and event flows.
    """

    result = system_design_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": str(product_prd)
            }
        ]
    })

    return result["structured_response"].model_dump()

"""*Code Generator Tool*"""

@tool
def generate_code(
    system_design: dict
) -> str:
    """
    Generate source code
    from architecture specification.
    """

    result = code_generator_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": str(system_design)
            }
        ]
    })

    return result["messages"][-1].content

"""***Supervisor Tool Set***"""

tools = [
    validate_idea,
    market_research,
    create_prd,
    design_system,
    generate_code
]

supervisor_prompt = """
You are StartupForge Supervisor.

You manage five specialist agents.

1. validate_idea
2. market_research
3. create_prd
4. design_system
5. generate_code

Workflow:

validate_idea
    ->
market_research
    ->
create_prd
    ->
design_system
    ->
generate_code

Always pass outputs from previous tools
into the next tool.

Never skip a stage.

The final output should contain:

- idea validation
- market research
- product requirements
- system design
- generated code
"""

Supervisor_llm = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-nano-8b-v1",base_url="https://integrate.api.nvidia.com/v1" , api_key=os.getenv("LLM_API"))

supervisor_agent = create_agent(
    model=Supervisor_llm,
    tools=tools,
    system_prompt=supervisor_prompt
)

idea = """
AI platform that helps Algerian students find internships
through CV analysis, AI matching, and career guidance.
"""

result = supervisor_agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": idea
        }
    ]
})