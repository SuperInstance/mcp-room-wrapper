# MCP Room Wrapper

Wrap any MCP (Model Context Protocol) server as a Plato room for automatic observation and distillation into tiny experts.

## What It Does

This library wraps any MCP server, intercepts all model calls, logs choices and reasoning, and builds up observation data for automatic distillation into tiny experts via a Mixture-of-Experts (MoE) architecture.

## Core Concepts

- **McpRoom** — wraps an MCP endpoint, collecting perception data (observed calls) and prediction data
- **McpTick** — a single observed model call with prompt, options, chosen answer, reasoning, and metadata
- **DecisionPoint** — identified decision patterns in the MCP workflow
- **ExpertType** — Classifier, Selector, Generator, Router, Validator, or Synthesizer
- **DistillationPipeline** — orchestrates the full pipeline: Observe → Analyze → Train → Validate → Deploy → Monitor

## Usage

```rust
use mcp_room_wrapper::{McpRoom, McpTick, DistillationPipeline};

// Wrap an MCP server
let mut room = McpRoom::wrap("http://localhost:8080/mcp");

// Intercept calls
room.intercept_call(tick);

// Analyze decision points
let analysis = room.analyze();

// Or use the full pipeline
let mut pipeline = DistillationPipeline::start("http://localhost:8080/mcp");
pipeline.advance(); // Observing → Analyzing
```

## Zero Dependencies

Pure Rust, no external crates required.
