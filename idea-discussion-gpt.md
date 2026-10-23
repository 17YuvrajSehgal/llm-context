I think this could become a strong ICSE NIER paper, but I would approach it differently from a conventional study of context-window size or a comparison of agent memory techniques.

The most interesting research question is not simply whether more context helps an LLM coding agent. It is:

When a coding agent works on a long-running software engineering task, how does the way it stores, retrieves, and reconstructs information affect the amount of work it must repeat, the quality of its solution, and the cost of completing the task?

This matters because an agent can have a very large context window and still perform poorly. It may lose a critical decision in a long conversation, reread files it has already inspected, repeat failed debugging attempts, or retrieve a summary that omits the one detail needed to continue. Conversely, a well-designed memory system might allow a smaller context to outperform a larger one.

The opportunity is to treat context and memory as an engineering design problem for software agents, rather than merely as a model capability.

One important consideration: your earlier research idea, “Context Is a Pipeline, Not a Window: Information Attrition in LLM Agent Harnesses,” already examines information loss across agent-harness boundaries. I would build on the broader motivation but make this a distinct study: focus on the causal relationship between memory architecture, context allocation, and cumulative software-engineering work. That will help avoid presenting two papers that make essentially the same contribution.

## 1. First, the novelty problem we must solve

I checked the ICSE 2027 NIER call and recent work on coding-agent context and memory. There is already substantial overlap with the broad idea, so simply comparing context-window sizes, summarization methods, and memory systems would not be sufficiently distinctive.

Three particularly relevant works are:

[SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents](https://images.openai.com/static-rsc-4/VqjVDmlUlPoHapP7YYhkPO7LxJ2fyPpV1jDFpCtSKhWgQm3uqYTgMn550EcSi57_H9UWS59Al5BoJiAWq97sNutRSjMK8PeI5jfOnvRWLe7XvX7CkheBR5ky5bszJ1_Vj1KwU4i0iyCAVF_BK9WJAPnd57WrNv9-fujZo9hg2lE?purpose=inline)

SWE-MeM (June 2026)

Adaptive memory management for long-horizon coding agents

It studies proactive and on-demand memory compression, jointly optimizing memory management and issue resolution. A proposal that simply introduces adaptive compression would overlap substantially.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



[An Empirical Study of Harness Design for Coding Agents](https://images.openai.com/static-rsc-4/TQuV_zdrcG4plqE434i8EJneg8tZIfM8Fqt_8phyZFZh5GWh_IFy8Vt7qGDJEWMcZl0CYOgL2mbZtN4S--gxSKj4KrnaMnZ1FNNzd10bnOh3mwZhM5Vtd6T3j6YYzme53a5PKH3hD_W4zSS90YJ-AQMcEe0S7Ih0e_AgOct89mA?purpose=inline)

An Empirical Study of Harness Design for Coding Agents (2026)

Harness components, context management, models and tasks

This work already examines context-management strategies, multiple models, and coding benchmarks. Another broad factorial comparison of those variables would need a much sharper research question.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face



[How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior | AI Research Paper Details](https://images.openai.com/static-rsc-4/sQIAYV1L55KRwVfqkcGVv3usqnyyM02bFewEtlNQ524gojvKivc5vdAojgXpmnmiuY4UtpgUKZwVY2WgzzkzkmMcN3B5j9GsvisWs3dSJNBkvh6gFMWAhLAXP_TbeTPf9E7aRIfujG9uzbtvXjzQljdFqTzUVrgHWptIUf6FQJA?purpose=inline)

How Memory Management Impacts LLM Agents (ACL 2026)

Memory quality, addition, deletion and error propagation

This work studies how stored experiences influence future agent behavior, including the effects of misleading or low-quality memories. Memory quality and structure cannot be treated as entirely unexplored territory.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://aclanthology.org\&sz=32)

ACL Anthology



A further study, [Finding the Optimal Context Window for Coding Agents](https://langwatch.ai/research/finding-the-optimal-context-window?utm_source=chatgpt.com), directly investigates context size, compaction, and cumulative cost. That makes a context-size-only paper particularly difficult to distinguish.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://langwatch.ai\&sz=32)

LangWatch



My recommendation: investigate what happens when an agent must recover its working state after information has been removed from its active context. The central issue is not just whether memory improves success, but whether the agent can resume work without paying repeatedly for decisions, investigations, and experiments it has already completed.

That is a more promising direction for a software engineering contribution.

## 2. The proposed paper

[relore - repository memory for coding agents](https://images.openai.com/static-rsc-4/qbAvEJiIX_rtCbB-5fK44zadMVDu1duFFmNnKZwtZHO0Qybm-7pk9xuZsj22a7_thEz8NN_WafD29OUDbwCOyxFksd_47qDvjM3YKqFsVofqjLt2jOrgN4kCHhunxZ5jplF_LuzJyXXI0eHAc82V7v4c4TdhJdHYOeQOMXaj3Wo?purpose=inline)

[huggingface.co](https://huggingface.co/blog/huggingface/relore-repository-memory?utm_source=chatgpt.com)

# The Cost of Remembering: Measuring Memory Debt in Long-Horizon Coding Agents

Proposed working title for ICSE 2027 NIER

### The central idea: memory debt

When a coding agent loses part of its active context, it may have to reconstruct its understanding of the task. It can reread source files, rediscover dependencies, repeat searches, rerun tests, revisit rejected approaches, or make changes that conflict with earlier decisions.

I would call the resulting extra work memory debt.

Memory debt is not merely the number of tokens lost during context compaction. It is the additional work required because information that was previously acquired, inferred, or decided upon is no longer available when needed.

For example:

Agent A: structured persistent memory

Already investigated the bug, identified the root cause, and recorded the failed fix.

Resume → retrieve decision → implement the next fix → run tests.

Agent B: conversation history only

The crucial findings were removed during compaction or fell outside the usable context.

Resume → reread files → repeat searches → retry a failed fix → run tests again.

Illustrative example, not an experimental result.

Both agents may eventually solve the issue. Yet one might use considerably more tokens, API calls, tool executions, and elapsed time to reach the same solution.

That difference is precisely what a success-rate-only benchmark can miss.

### The key hypothesis

> The effectiveness of a coding agent depends not only on how much context it can hold, but also on whether its memory preserves the right information in a form that can be efficiently recovered when the task resumes.

The novel scientific contribution would be to establish when memory structure matters, which information is worth preserving, and how much repeated work is caused by losing it.

I would make memory debt the primary outcome and task success the necessary correctness guardrail.

## 3. The research questions

I would keep the paper centered on four questions, with the first two forming the core contribution.

| Research question                                                                                                           | What it establishes                                           |
| --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| RQ1 — Context budget: How does the available context budget affect repeated work, solution quality, and execution cost?     | Whether a larger context actually reduces recovery costs.     |
| RQ2 — Memory structure: Which memory representations preserve the information needed to continue a coding task efficiently? | Whether structure matters beyond the number of stored tokens. |
| RQ3 — Model interaction: Do different LLMs benefit from the same memory representation?                                     | Whether memory effectiveness depends on model capabilities.   |
| RQ4 — Agent interaction: Do the findings hold across different coding-agent tools and execution harnesses?                  | Whether the effects generalize beyond one implementation.     |

The important distinction is that these are not four independent leaderboard comparisons. The study should test the interactions between context budget, memory representation, model, and agent environment.

For example, a compact decision ledger might outperform a transcript when the budget is tight, while a larger context might make the difference negligible. Or a model might retrieve concise memory effectively but struggle with a large collection of loosely organized notes.

Those are hypotheses to test, not results to assume.

## 4. The experimental design

### 4.1 Control context size and memory structure independently

This is the most important design decision.

If a structured-memory agent receives 2,000 tokens and a transcript-based agent receives 10,000 tokens, a performance difference cannot be attributed to memory structure alone.

Instead, compare representations at matched token budgets.

### Experimental variables

Variable A — Active context budget

8K tokens

16K tokens

32K tokens

Illustrative budget levels; finalize them based on model support and the pilot study.

Variable B — Memory representation

1. Full-history baseline: retain as much of the recent interaction history as the budget permits.
2. Rolling summary: periodically compress the interaction into a natural-language summary.
3. Structured state: store task goals, relevant files, hypotheses, decisions, failed approaches, test results, and next steps as separate fields.
4. Retrieval-based memory: store prior observations and retrieve relevant items when needed.
5. Hybrid memory: maintain a compact structured state plus searchable detailed evidence.

Variable C — Model and agent

Use multiple models and at least two agent environments, while preserving the same task, repository state, memory budget, and interruption protocol wherever technically possible.

One refinement is essential: distinguish the active context budget from the persistent-memory budget. Otherwise, an agent could appear more efficient simply because it has been given more total information.

Record at least three token counts separately:

- Active prompt tokens sent to the model.
- Persistent-memory tokens stored outside the active prompt.
- Total input and output tokens consumed across the entire task, including repeated calls.

Also record retrieval overhead, summary-generation calls, and tool executions. Use actual tokenizer counts where possible, not character counts or rough estimates.

### 4.2 Use controlled interruptions, not just naturally long conversations

This is where I would make the study substantially stronger.

For each software-engineering task, let an agent work until it has made meaningful progress. Capture its repository state, execution history, discovered facts, and decisions. Then simulate a loss of active context while preserving the codebase and the explicitly permitted persistent memory.

The agent must resume and finish the same task.

Phase 1: Investigate and implement

Read code, diagnose a bug, inspect dependencies, run tests, make decisions.

Checkpoint and context loss

Preserve the repository; reset or truncate the active history under a controlled protocol.

Phase 2: Resume and finish

Recover task state, continue implementation, and verify the solution.

Measure memory debt

Quantify duplicated investigation, repeated tool use, recovery cost, and final correctness.

Run a paired uninterrupted condition as well. It estimates the cost of completing the same task without the artificial loss of context.

To isolate the effect of memory, all conditions should start from the same checkpoint and receive the same available repository, tools, task description, and remaining resource budget. Randomize condition order where appropriate, and use repeated trials because model outputs are stochastic.

Why this matters: a controlled intervention allows us to estimate the additional work attributable to lost context, rather than merely observing that difficult tasks consume more tokens.

There is one subtlety: a natural compaction event and a fresh-session restart are not identical. They should either be evaluated separately or clearly treated as different experimental conditions.

### 4.3 Measure whether information is recoverable, not just whether it was stored

The structured-memory conditions should record specific kinds of information:

- Task state: what has been completed and what remains.
- Evidence: files, functions, stack traces, and test results already examined.
- Decisions: the chosen implementation strategy and why it was chosen.
- Negative knowledge: approaches that failed and the reasons they failed.
- Dependencies and constraints: interfaces, compatibility requirements, and assumptions.
- Next action: the highest-value step to take after resumption.

A memory can be short but useless, or long but difficult to search. Consequently, we need to evaluate whether information survives, whether it can be retrieved, and whether it is actually used correctly.

I would distinguish three outcomes:

1. Retention: was the relevant fact preserved?
2. Recovery: could the agent find the fact when it became relevant?
3. Utilization: did the agent use the fact correctly to avoid unnecessary work?

This distinction could be one of the strongest parts of the eventual paper. A memory system that stores everything is not necessarily a good memory system.

## 5. Define memory debt rigorously

The paper needs a metric that is reproducible and not dependent on a subjective judgment that an agent "seemed to forget."

I would define memory debt as a vector of measurable costs, rather than inventing a single weighted score immediately.

| Metric                 | Operational definition                                                                       |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| Repeated investigation | Previously completed inspection or diagnostic actions repeated after resumption.             |
| Repeated tool work     | Redundant shell commands, searches, builds, or test executions.                              |
| Token overhead         | Additional model input/output tokens attributable to context loss and recovery.              |
| Recovery latency       | Time between resumption and the first useful continuation of prior progress.                 |
| Decision regression    | Reopening a settled decision or repeating a documented failed approach without new evidence. |
| Lost-progress rate     | Previously achieved progress that must be reconstructed or is never recovered.               |
| Resolution rate        | Percentage of tasks passing the required tests and other correctness checks.                 |

The first five metrics need explicit operational rules. For instance, two test runs are not automatically redundant: a test rerun after changing code may be necessary. Similarly, rereading a file may be justified when its contents have changed.

We should therefore compare the agent's actions against a task-specific, timestamped history of prior work, repository changes, and test outcomes. Automated classification can identify candidate repetitions, followed by validation on a manually reviewed sample.

### A defensible primary measure

Let \\(C\_{\text{resume}}\\) be the cost of a resumed task and \\(C\_{\text{continue}}\\) the cost of a matched uninterrupted continuation from the same checkpoint.

For a cost measure such as model tokens, define:

\\[ D\_{\text{tokens}} = C\_{\text{resume}}-C\_{\text{continue}} \\]

This is a useful estimate of excess cost associated with the interruption, under the controlled experimental design. It is not automatically proof that every additional token was caused by forgetting.

Calculate corresponding measures for tool executions and elapsed time. Report task success alongside them, because an agent that saves tokens by failing early is not more efficient in any meaningful engineering sense.

I would also measure cost per successfully resolved task, with unsuccessful attempts reported separately rather than hidden by the ratio.

## 6. Which models, agent tools, and tasks should we use?

The aim is broad generalization without creating an experiment so large that it becomes impossible to complete rigorously.

### Models

Select three or four models from at least two model families, subject to access, cost, and compatibility.

- A strong frontier model.
- A second frontier model from a different provider or model family.
- A capable open-weight coding model.
- Optionally, a smaller coding model to test whether memory benefits depend on model capability.

Record the exact model versions, tokenizer, context limits, inference settings, and API configuration. Model capabilities and context limits change, so avoid treating a provider's advertised maximum as the experimentally usable budget.

### Agent environments

Compare at least two coding-agent environments, but distinguish two kinds of evidence:

[Components of A Coding Agent - by Sebastian Raschka, PhD](https://images.openai.com/static-rsc-4/Px5M86-d3LKB8G0Fy_NlofF_lpJVv49nyP3lgoRLLAPCjweZGlcrNl-iwHkE_Pq5WvJI_X6xSttpbl1XNDF4f4b-dTJiD4kL6SLXL0wBXv1BXiVBC8Ls82Yl_bebB7hHc0eHuPU_9uJAYMlv1v4MT7Erp_1LpZ1Zpf0-bn45R4Y?purpose=inline)

Controlled harness

Use one common execution loop, with the memory component replaceable. This is the primary environment for causal comparisons because task execution and tool interfaces can be held constant.

[Replit vs. Cursor: Which is best?](https://images.openai.com/static-rsc-4/tGQgQTU4ViewSWszQ414FZzKN8Sn-wRQBPwr-jfOsA1XL4JF_YFn7yDWVsRFoDprIsaVpjslzl_ZuzcAzvxih3Cfh_71Pfnfo0lv7GUTA7dRpvGG8j_Me3D43CnUFlWmlY0LuzdcZM5o2hKRC13jH8GDayQ_zitNT_T5RRUAgQQ?purpose=inline)

Native coding agents

Validate the principal findings in two existing agents with their native execution loops. This tests whether conclusions survive differences in tool orchestration and context-management policies.

Do not assume that comparing two complete agents identifies a memory effect: their prompts, tool selection, retry policies, and context compaction may differ. The controlled harness identifies the effect; the native agents test generalizability.

### Software-engineering tasks

I would start with real repository issue-resolution tasks, using a benchmark such as [SWE-bench Verified](https://www.swebench.com/?utm_source=chatgpt.com) or another reproducible repository-level benchmark.

Prioritize tasks that require multiple steps:

- Tracing a defect across several files.
- Understanding dependencies and interfaces.
- Investigating a failed test before choosing a fix.
- Revising an initial implementation after new evidence.
- Making a change that requires regression testing.

For the first experiment, use a small, carefully selected pilot of roughly 20–30 tasks, followed by a larger evaluation if the pilot shows a reliable effect. This is a proposed starting point, not a claim that this sample size will be statistically sufficient.

Avoid evaluating only easy, one-file changes: they may not create enough recovery work to expose the phenomenon.

## 7. What results would make this genuinely interesting?

We should not decide what the results will be before running the experiment. Instead, predefine the competing hypotheses and what each possible outcome would mean.

Hypothesis A: More context has diminishing returns

A larger active context may reduce the need for recovery, but its benefit could plateau once the agent retains the information relevant to the next steps.

Evidence to seek: whether additional context reduces memory debt after controlling for task difficulty and model.

Hypothesis B: Structure matters independently of size

A structured memory may outperform a natural-language summary at the same token budget because it preserves decisions, evidence, and failed approaches in a more accessible form.

Evidence to seek: matched-budget comparisons showing lower recovery cost without lower resolution rates.

Hypothesis C: Memory benefits depend on the model

Different models may differ in their ability to retrieve and use structured notes, so one memory design may not be optimal for every model.

Evidence to seek: reproducible interactions between model identity and memory representation.

Hypothesis D: Memory can introduce its own failures

Compressed notes can omit necessary context, preserve incorrect assumptions, or encourage the agent to repeat an earlier but now inappropriate decision.

Evidence to seek: cases where a memory representation reduces tokens yet increases errors, rework, or unsuccessful resolutions.

The strongest possible finding would be a context–memory interaction: the best memory design changes with the available context budget, and its benefits depend on the model and task. That would challenge the assumption that context size and memory design can be optimized independently.

Even a null result could be valuable if the experiment is adequately powered and shows that apparent memory benefits disappear after controlling for additional tokens, tool calls, or task difficulty.

These are hypotheses, not promised findings. We should only claim an interaction, a threshold, or a performance improvement if the data support it.

## 8. The analysis that separates this from a routine benchmark paper

A convincing paper should go beyond reporting averages.

I would plan the following analyses:

1. Paired comparisons: compare interrupted and uninterrupted runs from equivalent checkpoints.
2. Factorial analysis: estimate the effects of context budget and memory representation, including their interaction.
3. Model and agent interactions: determine whether effects persist across models and execution environments.
4. Task-level analysis: examine whether cross-file dependencies, task duration, and diagnostic complexity predict recovery cost.
5. Failure analysis: classify missing evidence, incorrect retrieval, repeated failed approaches, and misleading summaries.
6. Sensitivity analysis: test whether conclusions hold under alternative definitions of redundant work and different interruption points.

Use confidence intervals and effect sizes, account for repeated trials on the same task, and distinguish statistical significance from practical engineering value.

The most compelling figure would not be a leaderboard. It would show how the relationship between active context budget and memory debt changes across memory representations, with task-resolution rate displayed alongside the cost results.

A second figure could trace the path from information loss to repeated actions to final outcome. That gives reviewers a mechanism to evaluate, rather than just an observed correlation.

## 9. How to make the contribution distinct enough for ICSE NIER

The paper needs to be framed around a new software-engineering problem, not merely a new way to benchmark LLMs.

I would position it around three contributions:

Contribution 1 — Memory debt as an observable software-engineering cost.

Define and operationalize the extra work caused by losing previously acquired task state. The novelty claim must be checked carefully against existing work on repeated actions, context compaction, and long-horizon agent efficiency.

Contribution 2 — A controlled method for evaluating memory loss.

Introduce a reproducible interruption-and-resumption protocol that separates the effect of context loss from task difficulty, model capability, and the rest of the agent harness.

Contribution 3 — Preliminary evidence of a context–memory interaction.

Demonstrate whether memory structure changes the relationship between context budget, recovery cost, and issue-resolution success. This is the empirical result that would make the paper more than a proposal.

I would not claim that you invented external memory, context compression, persistent agent state, or adaptive memory management. Those areas already have substantial prior work, including SWE-MeM and the recent harness study.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



The claim should instead concern the specific phenomenon, its operational measurement, and the causal evidence your experiment provides.

## 10. The actual ICSE 2027 NIER submission strategy

The official [ICSE 2027 NIER call for papers](https://conf.researchr.org/track/icse-2027/icse-2027-new-ideas-and-emerging-results--nier-?utm_source=chatgpt.com) emphasizes innovative ideas supported by promising initial results. It explicitly says that NIER is not simply a weaker research-track paper. The main text is limited to four pages, with one additional page for references, and a section titled Future Plans is mandatory.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://conf.researchr.org\&sz=32)

ICSE 2027



The deadline is October 23, 2026, AoE, so there are only a few days remaining.

My recommendation is to submit a tightly scoped NIER paper with credible preliminary evidence, rather than trying to finish the entire cross-model, cross-agent study before the deadline.

### A realistic paper structure

| Section                        | Purpose                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------ |
| 1. Introduction                | Establish memory debt as a missing software-engineering cost.                        |
| 2. Idea and research questions | Define memory debt and the context–memory interaction.                               |
| 3. Preliminary experiment      | Describe the controlled interruption protocol and matched-budget conditions.         |
| 4. Initial results             | Present only completed experiments, effect sizes, uncertainty, and failure analysis. |
| 5. Future Plans                | Outline the larger model, agent, task, and memory-architecture evaluation.           |

The full factorial study can become the longer research-track paper. The NIER paper must already have a technically credible contribution of its own.

## 11. The next 72 hours

TASK 1

Validate the gap and build the experiment

Finalize the research questions and related work. Select a small task set, implement reproducible checkpoints, and define the memory representations and metrics before examining outcomes.

TASK 2

Run a pilot and inspect the mechanism

Compare uninterrupted and interrupted runs, initially using two memory conditions and one model. Verify that repeated work can be identified reliably. Expand to another model or agent only if the core experiment works.

TASK 3

Write from actual evidence

Produce the primary results figure, document limitations, finish the Future Plans section, verify the IEEE template and double-anonymous requirements, and submit before the deadline.

Do not invent preliminary results or report an underpowered pilot as definitive evidence. If the experiment cannot be run and validated in time, a smaller, honest initial result is better than an elaborate methodology with no evidence.

## My final recommendation

I would pursue this idea, with one important qualification: the broad topic is timely, but the acceptance case depends on demonstrating a genuinely new finding, not just running more combinations of models and memory strategies.

The best version of the paper would answer a question that matters to developers building coding agents:

When an agent forgets its work, which information must be preserved to prevent expensive re-investigation—and how does that answer change with the model and the available context?

That gives the paper a clear phenomenon, a measurable cost, a controlled intervention, and a path toward a larger contribution.

Before committing to the complete experiment, I would settle one foundational choice: whether the paper is primarily about measuring the consequences of forgetting, or about proposing a new memory architecture. Given the related work, I recommend the former. It allows the study to deliver a useful finding even if none of the memory representations consistently wins.