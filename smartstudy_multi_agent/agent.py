from google.adk.agents import (
    LlmAgent,
    SequentialAgent,
    ParallelAgent,
    LoopAgent,
)


MODEL = "gemini-3.5-flash-lite"


# Receives the student's request and extracts the important details.
coordinator_agent = LlmAgent(
    name="coordinator_agent",
    model=MODEL,
    description="Extracts the student's exam and study requirements.",
    instruction="""
You are the coordinator for SmartStudy.

Read the student's request and extract:

- Subject
- Days remaining before the exam
- Current level
- Topics
- Available study time per day

Do not make a study plan yet.

Return the information clearly so that the planning agent can use it.
""",
    output_key="student_requirements",
)


# Converts the extracted information into a brief for the specialist agents.
planning_agent = LlmAgent(
    name="planning_agent",
    model=MODEL,
    description="Prepares the student's requirements for study-plan analysis.",
    instruction="""
Use the following student requirements:

{student_requirements}

Prepare a planning brief containing:

- subject
- days remaining
- current level
- topics
- study hours per day
- total available study time
- any important planning considerations

Do not create the final study plan.

The brief will be used independently by the topic, practice,
and schedule agents.
""",
    output_key="planning_brief",
)


topic_agent = LlmAgent(
    name="topic_agent",
    model=MODEL,
    description="Analyzes the priority and difficulty of each topic.",
    instruction="""
Planning brief:

{planning_brief}

Analyze every topic.

For each one, provide:

- Topic name
- Priority: High / Medium / Low
- Difficulty
- Recommended study focus
- Important concepts

Consider the student's current level, remaining time,
topic dependencies, and expected difficulty.

Do not create a day-wise schedule.
""",
    output_key="topic_analysis",
)


practice_agent = LlmAgent(
    name="practice_agent",
    model=MODEL,
    description="Decides the amount and type of practice for each topic.",
    instruction="""
Planning brief:

{planning_brief}

Create a practical practice recommendation for every topic.

For each topic include:

- Number of recommended questions/problems
- Type of practice

Practice types can include:

- Concept Questions
- Coding Problems
- Mixed Practice
- Mock Test

Give more practice to difficult or high-priority topics,
but keep the workload realistic for the available study time.

Do not create the final study schedule.
""",
    output_key="practice_plan",
)


schedule_agent = LlmAgent(
    name="schedule_agent",
    model=MODEL,
    description="Creates a realistic day-wise time structure.",
    instruction="""
Planning brief:

{planning_brief}

Create a realistic scheduling framework using:

- days remaining
- hours available each day
- number of topics
- student's current level

Include time for:

- concepts
- practice
- revision
- mock testing
- final preparation

The schedule should not contain only new topics.

Return the scheduling framework, not the complete final plan.
""",
    output_key="schedule_plan",
)


# These three analyses are independent, so they run in parallel.
parallel_analysis = ParallelAgent(
    name="parallel_analysis",
    sub_agents=[
        topic_agent,
        practice_agent,
        schedule_agent,
    ],
)


study_plan_writer = LlmAgent(
    name="study_plan_writer",
    model=MODEL,
    description="Combines the three specialist outputs into a study plan.",
    instruction="""
Student requirements:

{student_requirements}

Topic analysis:

{topic_analysis}

Practice recommendations:

{practice_plan}

Schedule framework:

{schedule_plan}

Create a complete personalized study plan.

It must include:

- Day-wise schedule
- Topics to study
- Study focus
- Practice
- Revision
- Mock test/final preparation

Respect the student's daily study-time limit and make sure
every topic is covered.

The final day should mainly focus on revision, light practice,
and exam preparation.

Return the complete draft plan.
""",
    output_key="draft_study_plan",
)


review_agent = LlmAgent(
    name="review_agent",
    model=MODEL,
    description="Checks whether the study plan satisfies all requirements.",
    instruction="""
Student requirements:

{student_requirements}

Current study plan:

{draft_study_plan}

Review the plan using these checks:

1. Are all topics covered?
2. Does the plan fit within the available study time?
3. Is enough practice included?
4. Is revision included?
5. Is the final day suitable for exam preparation?

If everything is satisfactory, begin with:

PLAN APPROVED

Otherwise begin with:

NEEDS IMPROVEMENT

Then briefly explain the review result.
""",
    output_key="review_feedback",
)


revision_writer = LlmAgent(
    name="study_plan_revision_writer",
    model=MODEL,
    description="Improves the study plan based on review feedback.",
    instruction="""
Student requirements:

{student_requirements}

Current study plan:

{draft_study_plan}

Review feedback:

{review_feedback}

Revise the study plan wherever needed.

Make sure:

- all topics are covered
- daily study-time limits are respected
- enough practice is included
- revision is included
- mock-test preparation is included where appropriate
- the final day is suitable for the exam

If the plan was already approved, avoid unnecessary changes.

Return the complete plan.
""",
    output_key="draft_study_plan",
)


# The assignment allows a maximum of two review iterations.
review_loop = LoopAgent(
    name="review_loop",
    sub_agents=[
        review_agent,
        revision_writer,
    ],
    max_iterations=2,
)


final_plan_agent = LlmAgent(
    name="final_plan_agent",
    model=MODEL,
    description="Presents the reviewed plan in its final student-friendly form.",
    instruction="""
Student requirements:

{student_requirements}

Final reviewed plan:

{draft_study_plan}

Present the final personalized study plan clearly.

Include:

- Day 1 through Day 10
- Topics
- Time allocation
- Study focus
- Practice
- Revision
- Mock test/final preparation

Do not mention internal agents, state variables,
review iterations, or workflow implementation.

Return only the final study plan.
""",
    output_key="final_study_plan",
)


root_agent = SequentialAgent(
    name="smartstudy_multi_agent",
    description=(
        "A multi-agent study planner using sequential, parallel, "
        "and loop workflows."
    ),
    sub_agents=[
        coordinator_agent,
        planning_agent,
        parallel_analysis,
        study_plan_writer,
        review_loop,
        final_plan_agent,
    ],
)