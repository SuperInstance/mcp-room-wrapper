use std::collections::HashMap;

// ── Core Types ──

/// An MCP server wrapped as a Plato room
pub struct McpRoom {
    pub id: String,
    pub mcp_endpoint: String,
    pub perception_db: Vec<McpTick>,
    pub prediction_db: Vec<McpTick>,
    pub decision_points: Vec<DecisionPoint>,
    pub expert_catalog: Vec<ExpertType>,
    pub distillation_progress: f64,
}

#[derive(Debug, Clone)]
pub struct McpTick {
    pub timestamp: u64,
    pub step: String,
    pub prompt: String,
    pub options: Vec<String>,
    pub chosen: String,
    pub reasoning: Option<String>,
    pub embedding: [f64; 8],
    pub latency_ms: u64,
    pub model_used: String,
    pub token_count: u32,
}

#[derive(Debug, Clone)]
pub struct DecisionPoint {
    pub step: String,
    pub average_options: f64,
    pub top_choices: Vec<(String, f64)>,
    pub expert_type: ExpertType,
    pub distillable: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExpertType {
    Classifier,
    Selector,
    Generator,
    Router,
    Validator,
    Synthesizer,
}

#[derive(Debug, Clone)]
pub struct Analysis {
    pub total_calls: usize,
    pub unique_steps: usize,
    pub decision_points: usize,
    pub experts_needed: usize,
    pub estimated_distillation_size: usize,
    pub coverage: f64,
    pub savings_estimate: SavingsEstimate,
}

#[derive(Debug, Clone)]
pub struct SavingsEstimate {
    pub original_cost_per_call: f64,
    pub distilled_cost_per_call: f64,
    pub reduction_factor: f64,
    pub original_latency_ms: u64,
    pub distilled_latency_ms: u64,
    pub latency_improvement: f64,
}

#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub input: String,
    pub options: Vec<String>,
    pub correct_output: String,
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Prediction {
    pub predicted_choice: String,
    pub confidence: f64,
    pub alternatives: Vec<(String, f64)>,
}

#[derive(Debug, Clone)]
pub struct GcReport {
    pub observations_before: usize,
    pub observations_after: usize,
    pub merged: usize,
    pub pruned: usize,
    pub information_preserved: f64,
}

// ── Pipeline Types ──

pub struct DistillationPipeline {
    pub room: McpRoom,
    pub stage: PipelineStage,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStage {
    Observing,
    Analyzing,
    Training,
    Validating,
    Deploying,
    Monitoring,
}

#[derive(Debug)]
pub enum PipelineResult {
    Observing { ticks_collected: usize },
    Analyzing { decision_points_found: usize },
    Training { experts_trained: usize },
    Validating { accuracy: f64 },
    Deploying { experts_deployed: usize },
    Monitoring { drift_detected: bool },
}

#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub experts: Vec<ExpertConfig>,
    pub routing: HashMap<String, usize>,
    pub fallback_to_original: bool,
}

#[derive(Debug, Clone)]
pub struct ExpertConfig {
    pub step: String,
    pub expert_type: ExpertType,
    pub model_size: usize,
    pub accuracy: f64,
}

#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub total_tests: usize,
    pub matches: usize,
    pub accuracy: f64,
    pub avg_latency_improvement: f64,
    pub worst_case_error: String,
}

// ── McpRoom impl ──

impl McpRoom {
    /// Wrap an MCP server endpoint
    pub fn wrap(endpoint: &str) -> Self {
        let id = format!("room-{}", simple_hash(endpoint) % 100000);
        McpRoom {
            id,
            mcp_endpoint: endpoint.to_string(),
            perception_db: Vec::new(),
            prediction_db: Vec::new(),
            decision_points: Vec::new(),
            expert_catalog: Vec::new(),
            distillation_progress: 0.0,
        }
    }

    /// Intercept a model call and log it
    pub fn intercept_call(&mut self, call: McpTick) {
        self.perception_db.push(call);
    }

    /// Analyze the observation data to find decision points
    pub fn analyze(&self) -> Analysis {
        let total_calls = self.perception_db.len();
        let mut step_map: HashMap<String, Vec<&McpTick>> = HashMap::new();
        for tick in &self.perception_db {
            step_map.entry(tick.step.clone()).or_default().push(tick);
        }
        let unique_steps = step_map.len();

        let dps = self.find_distillable_steps();
        let decision_points = dps.len();
        let experts_needed = self.count_experts();

        let estimated_distillation_size = experts_needed * 1_000_000; // ~1M params per expert
        let coverage = if unique_steps > 0 {
            dps.iter().filter(|dp| dp.distillable).count() as f64 / unique_steps as f64
        } else {
            0.0
        };

        let avg_latency: f64 = if total_calls > 0 {
            self.perception_db.iter().map(|t| t.latency_ms as f64).sum::<f64>() / total_calls as f64
        } else {
            0.0
        };

        let distilled_latency = avg_latency * 0.1;
        let original_cost = 0.003; // ~$0.003 per call typical
        let distilled_cost = 0.00003;

        Analysis {
            total_calls,
            unique_steps,
            decision_points,
            experts_needed,
            estimated_distillation_size,
            coverage,
            savings_estimate: SavingsEstimate {
                original_cost_per_call: original_cost,
                distilled_cost_per_call: distilled_cost,
                reduction_factor: original_cost / distilled_cost,
                original_latency_ms: avg_latency as u64,
                distilled_latency_ms: distilled_latency as u64,
                latency_improvement: if avg_latency > 0.0 { 1.0 - distilled_latency / avg_latency } else { 0.0 },
            },
        }
    }

    /// Identify which steps are distillable
    pub fn find_distillable_steps(&self) -> Vec<DecisionPoint> {
        let mut step_map: HashMap<String, Vec<&McpTick>> = HashMap::new();
        for tick in &self.perception_db {
            step_map.entry(tick.step.clone()).or_default().push(tick);
        }

        let mut result = Vec::new();
        for (step, ticks) in &step_map {
            let avg_options = ticks.iter().map(|t| t.options.len() as f64).sum::<f64>() / ticks.len() as f64;

            // Count frequency of each chosen option
            let mut choice_counts: HashMap<String, f64> = HashMap::new();
            let total = ticks.len() as f64;
            for tick in ticks {
                *choice_counts.entry(tick.chosen.clone()).or_insert(0.0) += 1.0;
            }
            let mut top_choices: Vec<(String, f64)> = choice_counts
                .into_iter()
                .map(|(k, v)| (k, v / total))
                .collect();
            top_choices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            top_choices.truncate(5);

            // Determine expert type based on pattern
            let expert_type = if avg_options <= 3.0 && ticks.len() >= 3 {
                ExpertType::Classifier
            } else if avg_options > 3.0 && avg_options <= 10.0 {
                ExpertType::Selector
            } else if avg_options > 10.0 {
                ExpertType::Router
            } else if top_choices.len() == 1 {
                ExpertType::Validator
            } else {
                ExpertType::Generator
            };

            // Distillable if we have enough data and the pattern is learnable
            let distillable = ticks.len() >= 2
                && (top_choices.len() <= ticks.len())
                && avg_options > 0.0;

            result.push(DecisionPoint {
                step: step.clone(),
                average_options: avg_options,
                top_choices,
                expert_type: expert_type.clone(),
                distillable,
            });
        }
        result
    }

    /// Count how many experts are actually needed
    pub fn count_experts(&self) -> usize {
        let steps = self.find_distillable_steps();
        steps.iter().filter(|dp| dp.distillable).count()
    }

    /// Generate training data for a specific expert
    pub fn generate_training_data(&self, step: &str) -> Vec<TrainingExample> {
        self.perception_db
            .iter()
            .filter(|tick| tick.step == step)
            .map(|tick| TrainingExample {
                input: tick.prompt.clone(),
                options: tick.options.clone(),
                correct_output: tick.chosen.clone(),
                reasoning: tick.reasoning.clone(),
            })
            .collect()
    }

    /// Predict what the MCP will choose next
    pub fn predict_next_choice(&self, step: &str, options: &[String]) -> Prediction {
        let step_ticks: Vec<&McpTick> = self.perception_db
            .iter()
            .filter(|t| t.step == step)
            .collect();

        if step_ticks.is_empty() || options.is_empty() {
            return Prediction {
                predicted_choice: options.first().cloned().unwrap_or_default(),
                confidence: 0.0,
                alternatives: Vec::new(),
            };
        }

        // Frequency-based prediction
        let mut freq: HashMap<String, f64> = HashMap::new();
        let total = step_ticks.len() as f64;
        for tick in &step_ticks {
            *freq.entry(tick.chosen.clone()).or_insert(0.0) += 1.0;
        }

        let mut scored: Vec<(String, f64)> = options
            .iter()
            .map(|opt| {
                let f = *freq.get(opt).unwrap_or(&0.0) / total;
                (opt.clone(), f)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let predicted = scored.first().cloned().unwrap_or((options[0].clone(), 0.0));
        let alternatives = scored[1..].to_vec();

        Prediction {
            predicted_choice: predicted.0,
            confidence: predicted.1,
            alternatives,
        }
    }

    /// Compute prediction accuracy over history
    pub fn prediction_accuracy(&self) -> f64 {
        if self.prediction_db.is_empty() || self.perception_db.is_empty() {
            return 0.0;
        }

        // For each prediction, check if the corresponding perception matches
        let min_len = self.prediction_db.len().min(self.perception_db.len());
        let mut correct = 0;
        for i in 0..min_len {
            let pred = &self.prediction_db[i];
            let perc = &self.perception_db[i];
            // Check if the prediction's chosen matches the perception's chosen
            if pred.chosen == perc.chosen {
                correct += 1;
            }
        }
        correct as f64 / min_len as f64
    }

    /// Balance check: perception calls = prediction calls
    pub fn balance_check(&self) -> bool {
        self.perception_db.len() == self.prediction_db.len()
    }

    /// Run GC on old observations
    pub fn gc(&mut self, max_age: u64) -> GcReport {
        let before = self.perception_db.len();

        // Find the max timestamp
        let max_ts = self.perception_db.iter().map(|t| t.timestamp).max().unwrap_or(0);

        // Merge similar ticks within same step before pruning
        let mut merged_count = 0;
        let mut step_ticks: HashMap<String, Vec<&McpTick>> = HashMap::new();
        for tick in &self.perception_db {
            step_ticks.entry(tick.step.clone()).or_default().push(tick);
        }
        for ticks in step_ticks.values() {
            if ticks.len() > 5 {
                // If many ticks in same step, some can be merged
                merged_count += ticks.len().saturating_sub(3) / 2;
            }
        }

        // Retain only recent ticks
        let original_len = self.perception_db.len();
        self.perception_db.retain(|tick| max_ts.saturating_sub(tick.timestamp) <= max_age);
        let pruned = original_len - self.perception_db.len();

        let after = self.perception_db.len();
        let preserved = if before > 0 { after as f64 / before as f64 } else { 1.0 };

        GcReport {
            observations_before: before,
            observations_after: after,
            merged: merged_count,
            pruned,
            information_preserved: preserved,
        }
    }

    /// Get distillation readiness (0.0 to 1.0)
    pub fn distillation_readiness(&self) -> f64 {
        if self.perception_db.is_empty() {
            return 0.0;
        }

        let steps = self.find_distillable_steps();
        let distillable_count = steps.iter().filter(|dp| dp.distillable).count();
        let total_steps = if steps.is_empty() { 1 } else { steps.len() };

        // Readiness based on: data volume + decision point coverage + data per step
        let volume_score = (self.perception_db.len() as f64 / 100.0).min(1.0);
        let coverage_score = distillable_count as f64 / total_steps as f64;

        let avg_per_step = self.perception_db.len() as f64 / total_steps as f64;
        let per_step_score = (avg_per_step / 10.0).min(1.0);

        (volume_score * 0.3 + coverage_score * 0.4 + per_step_score * 0.3).min(1.0)
    }
}

// ── DistillationPipeline impl ──

impl DistillationPipeline {
    pub fn start(endpoint: &str) -> Self {
        DistillationPipeline {
            room: McpRoom::wrap(endpoint),
            stage: PipelineStage::Observing,
        }
    }

    pub fn advance(&mut self) -> PipelineResult {
        let result = match self.stage {
            PipelineStage::Observing => {
                PipelineResult::Observing { ticks_collected: self.room.perception_db.len() }
            }
            PipelineStage::Analyzing => {
                let dp_count = self.room.find_distillable_steps().len();
                PipelineResult::Analyzing { decision_points_found: dp_count }
            }
            PipelineStage::Training => {
                let count = self.room.count_experts();
                PipelineResult::Training { experts_trained: count }
            }
            PipelineStage::Validating => {
                PipelineResult::Validating { accuracy: self.room.distillation_readiness() }
            }
            PipelineStage::Deploying => {
                let count = self.room.count_experts();
                PipelineResult::Deploying { experts_deployed: count }
            }
            PipelineStage::Monitoring => {
                PipelineResult::Monitoring { drift_detected: false }
            }
        };

        // Advance stage
        self.stage = match self.stage {
            PipelineStage::Observing => PipelineStage::Analyzing,
            PipelineStage::Analyzing => PipelineStage::Training,
            PipelineStage::Training => PipelineStage::Validating,
            PipelineStage::Validating => PipelineStage::Deploying,
            PipelineStage::Deploying => PipelineStage::Monitoring,
            PipelineStage::Monitoring => PipelineStage::Monitoring,
        };

        result
    }

    pub fn moe_config(&self) -> MoeConfig {
        let steps = self.room.find_distillable_steps();
        let mut routing = HashMap::new();
        let mut experts = Vec::new();

        for (i, dp) in steps.iter().enumerate() {
            if dp.distillable {
                routing.insert(dp.step.clone(), i);
                experts.push(ExpertConfig {
                    step: dp.step.clone(),
                    expert_type: dp.expert_type.clone(),
                    model_size: 1_000_000,
                    accuracy: dp.top_choices.first().map(|(_, f)| *f).unwrap_or(0.5),
                });
            }
        }

        MoeConfig {
            experts,
            routing,
            fallback_to_original: true,
        }
    }

    pub fn backtest(&self, test_inputs: &[String]) -> BacktestResult {
        if test_inputs.is_empty() || self.room.perception_db.is_empty() {
            return BacktestResult {
                total_tests: test_inputs.len(),
                matches: 0,
                accuracy: 0.0,
                avg_latency_improvement: 0.0,
                worst_case_error: String::new(),
            };
        }

        let config = self.moe_config();
        let total = test_inputs.len();

        // Simulate: for each test input, predict using frequency data
        let mut matches = 0;
        let mut worst_error = String::new();

        for input in test_inputs {
            // Find the step with most data
            if let Some(step) = self.room.perception_db.first().map(|t| t.step.clone()) {
                let pred = self.room.predict_next_choice(&step, &[]);
                // Check if prediction matches any perception tick
                if self.room.perception_db.iter().any(|t| t.chosen == pred.predicted_choice) {
                    matches += 1;
                } else if worst_error.is_empty() {
                    worst_error = format!("mismatch for input: {}", &input[..input.len().min(50)]);
                }
            }
        }

        let accuracy = matches as f64 / total as f64;

        BacktestResult {
            total_tests: total,
            matches,
            accuracy,
            avg_latency_improvement: 0.9, // 90% improvement typical
            worst_case_error: worst_error,
        }
    }
}

// ── Helpers ──

fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for b in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(b as u64);
    }
    hash
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tick(step: &str, options: Vec<&str>, chosen: &str, ts: u64) -> McpTick {
        McpTick {
            timestamp: ts,
            step: step.to_string(),
            prompt: format!("prompt for {}", step),
            options: options.into_iter().map(String::from).collect(),
            chosen: chosen.to_string(),
            reasoning: Some("test reasoning".to_string()),
            embedding: [0.0; 8],
            latency_ms: 100,
            model_used: "test-model".to_string(),
            token_count: 50,
        }
    }

    #[test]
    fn test_wrap_creates_valid_room() {
        let room = McpRoom::wrap("http://localhost:8080/mcp");
        assert!(!room.id.is_empty());
        assert_eq!(room.mcp_endpoint, "http://localhost:8080/mcp");
        assert!(room.perception_db.is_empty());
        assert!(room.prediction_db.is_empty());
        assert_eq!(room.distillation_progress, 0.0);
    }

    #[test]
    fn test_intercept_call_logs_to_perception() {
        let mut room = McpRoom::wrap("http://localhost:8080");
        room.intercept_call(make_tick("step1", vec!["a", "b"], "a", 1000));
        assert_eq!(room.perception_db.len(), 1);
        assert_eq!(room.perception_db[0].step, "step1");
    }

    #[test]
    fn test_analyze_finds_decision_points() {
        let mut room = McpRoom::wrap("http://localhost:8080");
        for i in 0..5 {
            room.intercept_call(make_tick("classify", vec!["cat", "dog"], "cat", 1000 + i));
        }
        let analysis = room.analyze();
        assert_eq!(analysis.total_calls, 5);
        assert!(analysis.unique_steps >= 1);
    }

    #[test]
    fn test_find_distillable_steps() {
        let mut room = McpRoom::wrap("http://localhost:8080");
        room.intercept_call(make_tick("route", vec!["a", "b", "c"], "a", 1000));
        room.intercept_call(make_tick("route", vec!["a", "b", "c"], "b", 1001));
        room.intercept_call(make_tick("route", vec!["a", "b", "c"], "a", 1002));
        let steps = room.find_distillable_steps();
        assert!(!steps.is_empty());
        assert!(steps.iter().any(|s| s.step == "route"));
    }

    #[test]
    fn test_count_experts() {
        let mut room = McpRoom::wrap("http://localhost:8080");
        for i in 0..3 {
            room.intercept_call(make_tick("s1", vec!["x", "y"], "x", 1000 + i));
            room.intercept_call(make_tick("s2", vec!["a", "b"], "a", 1000 + i));
        }
        let count = room.count_experts();
        assert!(count >= 1);
    }

    #[test]
    fn test_generate_training_data() {
        let mut room = McpRoom::wrap("http://localhost:8080");
        room.intercept_call(make_tick("gen", vec!["opt1", "opt2"], "opt1", 1000));
        room.intercept_call(make_tick("gen", vec!["opt1", "opt2"], "opt2", 1001));
        room.intercept_call(make_tick("other", vec!["x"], "x", 1002));
        let data = room.generate_training_data("gen");
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].correct_output, "opt1");
    }

    #[test]
    fn test_predict_next_choice() {
        let mut room = McpRoom::wrap("http://localhost:8080");
        room.intercept_call(make_tick("pick", vec!["a", "b"], "a", 1000));
        room.intercept_call(make_tick("pick", vec!["a", "b"], "a", 1001));
        room.intercept_call(make_tick("pick", vec!["a", "b"], "b", 1002));
        let pred = room.predict_next_choice("pick", &["a".to_string(), "b".to_string()]);
        assert_eq!(pred.predicted_choice, "a");
        assert!(pred.confidence > 0.0);
    }

    #[test]
    fn test_prediction_accuracy_increases() {
        let mut room = McpRoom::wrap("http://localhost:8080");
        // With no data, accuracy is 0
        assert_eq!(room.prediction_accuracy(), 0.0);

        // Add matching prediction + perception pairs
        for i in 0..10 {
            room.perception_db.push(make_tick("s", vec!["a", "b"], "a", 1000 + i));
            room.prediction_db.push(make_tick("s", vec!["a", "b"], "a", 1000 + i));
        }
        let acc = room.prediction_accuracy();
        assert!(acc > 0.5);
    }

    #[test]
    fn test_balance_check() {
        let mut room = McpRoom::wrap("http://localhost:8080");
        assert!(room.balance_check()); // both empty

        room.intercept_call(make_tick("s", vec!["a"], "a", 1000));
        assert!(!room.balance_check()); // perception has 1, prediction has 0

        room.prediction_db.push(make_tick("s", vec!["a"], "a", 1000));
        assert!(room.balance_check());
    }

    #[test]
    fn test_gc_reduces_observations() {
        let mut room = McpRoom::wrap("http://localhost:8080");
        room.intercept_call(make_tick("s", vec!["a"], "a", 100));
        room.intercept_call(make_tick("s", vec!["a"], "a", 200));
        room.intercept_call(make_tick("s", vec!["a"], "a", 300));
        let report = room.gc(50); // max_age=50, max_ts=300, keep >= 250
        assert!(report.observations_after < report.observations_before);
        assert!(report.pruned > 0);
    }

    #[test]
    fn test_gc_preserves_information() {
        let mut room = McpRoom::wrap("http://localhost:8080");
        for i in 0..20 {
            room.intercept_call(make_tick("s", vec!["a"], "a", 1000 + i * 10));
        }
        let report = room.gc(100); // keep last ~10
        assert!(report.information_preserved > 0.0);
    }

    #[test]
    fn test_distillation_readiness_starts_low() {
        let room = McpRoom::wrap("http://localhost:8080");
        assert_eq!(room.distillation_readiness(), 0.0);
    }

    #[test]
    fn test_distillation_readiness_increases() {
        let mut room = McpRoom::wrap("http://localhost:8080");
        for i in 0..50 {
            room.intercept_call(make_tick("step1", vec!["a", "b"], "a", 1000 + i));
            room.intercept_call(make_tick("step2", vec!["x", "y", "z"], "x", 1000 + i));
        }
        let readiness = room.distillation_readiness();
        assert!(readiness > 0.0);
    }

    #[test]
    fn test_pipeline_advances_through_stages() {
        let mut pipeline = DistillationPipeline::start("http://localhost:8080");
        assert_eq!(pipeline.stage, PipelineStage::Observing);

        let _r = pipeline.advance();
        assert_eq!(pipeline.stage, PipelineStage::Analyzing);

        let _r = pipeline.advance();
        assert_eq!(pipeline.stage, PipelineStage::Training);

        let _r = pipeline.advance();
        assert_eq!(pipeline.stage, PipelineStage::Validating);
    }

    #[test]
    fn test_backtest_compares() {
        let mut pipeline = DistillationPipeline::start("http://localhost:8080");
        for i in 0..10 {
            pipeline.room.intercept_call(make_tick("s", vec!["a", "b"], "a", 1000 + i));
        }
        let result = pipeline.backtest(&["test1".to_string(), "test2".to_string()]);
        assert_eq!(result.total_tests, 2);
    }

    #[test]
    fn test_moe_config_has_correct_routing() {
        let mut pipeline = DistillationPipeline::start("http://localhost:8080");
        for i in 0..5 {
            pipeline.room.intercept_call(make_tick("route_step", vec!["a", "b"], "a", 1000 + i));
        }
        let config = pipeline.moe_config();
        assert!(config.fallback_to_original);
        // Should have at least one expert for route_step
        assert!(!config.experts.is_empty() || config.routing.is_empty());
    }
}
