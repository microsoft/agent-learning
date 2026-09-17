---
title: Agent Framework Integration
description: Architecture and implementation decisions for integrating agent-learning with Microsoft Agent Framework through middleware, policy-driven tools, and episode capture.
author: Microsoft
ms.date: 2026-09-16
ms.topic: concept
keywords:
  - agent framework
  - integration
estimated_reading_time: 5
---
# Agent framework integration

## Overview
Users of Microsoft Agent Framework can wrap any set of supported tools and attach an adapter to an agent to use agent-learning policies.

## API
Users create native agent-learning instances for:
- LearningStore
- LearningRunner

Users create Agent Framework tools, for example by using the `@tool` decorator on a Python function.

A learning adapter takes the Agent Framework tools, the learning store, and functions that derive decision frames, episode metadata, and other values from context available during function invocation:

```
learning = AgentFrameworkLearningAdapter.from_tools(
    agent_id=AGENT_ID,
    task_id=TASK_ID,
    action_tools=build_action_tools(catalog),
    ...
)
```

The adapter creates a low-authority policy by default. It loads the active policy if one exists.

The adapter is then registered with an Agent Framework agent:

```
learning.register(agent)
```

Registration sets up the agent to use agent-learning to make decisions, execute tools, and capture episodes in the provided store.

Users invoke Agent Framework's regular `.run()` method. That method can return approval requests as native Agent Framework objects:

```
response = await agent.run(prompt, session=session)
while response.user_input_requests:
    # ...
    # process request (prompt user)
    # ...
    response = await agent.run(
        request.to_function_approval_response(
            approved=True
        ),
        session=session
    )
```

After episode capture, users use the runner and store normally.

Policy updates are explicit and initiated by the user with:

```
learning.update_policy(policy.snapshot())
```

## Composite tool approach
The main implementation choice is to expose all tools belonging to a task ID and policy as a composite "meta" tool whose signature includes the inputs of all the underlying tools.

That composite tool interacts with agent-learning policies to resolve decisions and then execute selected tools. This means that the agent is not aware of the underlying tools. The composite tool is advertised as a single tool to the agent with its own name and description.

The composite tool sits alongside other regular tools the agent may have.

The advantage of this approach is that within one agent round trip, the agent decides to use the task and the action is selected and (with proper approvals) executed.

An alternative would be to model the task / policy as a separate tool and force selected tool calls as follow-up calls. However, Agent Framework has no built-in capability to force those calls nor correlate them properly with their initial decision.


## Approvals
Approvals and user requests use Agent Framework's built-in capabilities. When needed, the adapter returns a user input request from the composite tool call. Callers receive that request, prompt end users, and then send back an approval object. Both are built-in Agent Framework objects.

One challenge is retrieving past decision details upon receiving approval. In traditional Agent Framework approval flows, the tool call is replayed with the same arguments and unique call ID. Here, replaying the composite tool call would resolve the decision again. Instead, the adapter uses the user's session to track past decisions. For a given call to the composite tool, the adapter recognizes whether the call follows an approval and looks up the pre-approval decision details by using the unique call ID.

## Limitations
- Only one adapter can be added per agent to avoid state and middleware conflicts
- Managed agents (GitHub Copilot and Claude) with their own provider-managed tool loops are not supported
- Contextual softmax policies are not supported
- The model must provide inputs for every candidate action, even though only the selected action executes (useful for future support of contextual softmax policies)
