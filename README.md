# Multi-LoRAs System with LLM Core and Continuous Timestamped Training
Seems like global LLM models are no longer limited by model size, but by their structure. The goal here is to combat catastrophic forgetting with a local agent performing continuous timestamped learning, and that "**Caching is so last century, real memories ought to be stored directly in NN.**"

The general idea is to keep the central LLM unchanged, while surrounding it with a dozen LoRAs, each with an assigned weight and made to be orthogonal to each other. The system is made to be enclosed in an agent that does real-time observation, i.e., a robot, and perform continuous training. Every short moment, each LoRA will be queried with "Summarize your observation and your last output, and what are the times?" followed by the time information and the real-time data from the agent. At the same time, the LLM will be queried with "Summarize your observation and the most recent user input," followed by the real-time data from the agent and the user input (separating the user input from other observations, leaving room for reinforcement learning). A machine then updates the relevant weights between the outputs of the LoRAs and the LLM and trains on the LoRAs with the learning rate scaled by the weights. While using most of the core/RAM for training, the other core/RAM is used to continue generating output from LoRAs that include commands for the agent, e.g., move or reply back in voices.

The idea is that once the LoRAs reach a threshold number, say thousands, the multi-LoRA system will start having an emergent state towards general AGI, such as having long-term memory and planning.

## Concepts

### Real memory ought to be stored in the Neural Network

The idea basically originates from "You know, Neural Network weights can store everything we know of, why not use them to store memory instead of temporary KV-caching or external databases? After all, most human don't have photographic memory, but rather internalize their experiences." This is followed by a parallel speculation that a global entity will have a difficult time avoiding becoming just a giant dictionary; only local agents with real-time experience of their surroundings can develop self-awareness. 

Based on these two ideas, the proposal is to use the base LLM as a static core and LoRAs as dynamic "satellites" on an enclosed local agent with real-time observation using its sensors or other inputs. The system treats fine-tuning as the primary method for long-term memory storage. Sort of treating LLM as a permanent brain and LoRAs as the active "hippocampus." Then, by assigning specific weights to each LoRA, the system can balance between specializing and diversifying its knowledge sets. The most relevant LoRA to the observation will receive the highest priority in both learning and acting on the observation. Moreover, to prevent multiple LoRAs from learning the same information or "collapsing" into a single state, this approach also requires enforcing mathematical orthogonality between them. 

## Summarize your observation as a snapshot for memory

Agent-based observation

The system functions as an active agent (e.g., a robot) that constantly creates "snapshots" of its environment. By asking the LoRAs and the LLM to summarize current sensor data and their own recent outputs, the architecture converts raw real-time data into high-level, digestible reflections that are then used for training.

The agent is always watching. Every few moments, it summarizes its observations and recent actions into a "snapshot." These high-level summaries are what we feed back into the training loop, turning raw sensor data into meaningful experience.

## Real-time training demands timestamping your memory

LLM knowledge has no intrinsic time-ordering, this along will mean it will for every just be a giant dictionary

Standard LLMs lack an intrinsic sense of time, functioning more like static dictionaries than conscious entities. This subsection argues that by attaching timestamps to every training cycle, the agent learns the chronological order of events, transforming a collection of facts into a coherent, time-aware history of experiences.

LLMs are usually giant, static dictionaries with no sense of "before" or "after." By baking timestamps directly into the training data, we give the model a chronological sense of history. It learns the flow of time, not just a list of facts.

## Feedback loop for retrospection

User input for reinforcement learning

This component separates human user input from general environmental observations to create a dedicated channel for Reinforcement Learning (RL). By analyzing user reactions against its own previous actions, the agent can perform "retrospection," refining its behavior based on success or failure to better align with user expectations.

We treat user input as a special signal for Reinforcement Learning. The system looks back at its previous actions, compares them to user feedback, and adjusts. It’s not just acting; it’s reflecting on whether it did a good job.

## Goal and prospect 

OpenAI published a landmark paper showing that if you increase three things—Compute, Data Size, and Model Parameters—the loss (error rate) drops following a predictable power law.
The "Miracle" part: Unlike almost every other technology, where you hit "diminishing returns" quickly, Transformers showed that the more you fed the beast, the smarter it got, and we haven't hit the ceiling yet.

Structural upgrade

The next phase transition

Mixture of Experts Dynamically Trained LoRAs

Once we hit a critical mass of thousands of specialized LoRAs, we expect to see emergent AGI—long-term planning, deep memory, and a persistent "personality" that evolves every second.

The Multi-LoRA Hypothesis:
"We believe that the next Power Law isn't found in the parameter count of the LLM core, but in the LoRA Count. Just as the Transformer performance scaled with model size, Agent intelligence will scale with the number of orthogonal LoRA satellites. Once the system reaches a 'Critical Mass' (e.g., 1,000+ specialized LoRAs), we expect to see emergent General Intelligence—moving from a static dictionary to a time-aware, planning entity."

## Project

Basic plan, 3 Loras system on Qwen3 on an agent that can observe the clock, local weather report and GPU usage.
Need to figure out how to do weight + orthogonality

## References:
- John Schulman in collaboration with others at Thinking Machines, LoRA Without Regret (2025) (<a href="https://thinkingmachines.ai/blog/lora/">Blog</a>)
- 51616, Text-to-LoRA (T2L): Instant Transformer Adaption (LoRA Reinforcement Learning) (2025) (<a href="https://github.com/SakanaAI/text-to-lora">GitHub</a>)
