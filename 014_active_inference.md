# BRIEFING: ActiveInference

## Component ID: ORE2-014
## Priority: High (Makes ORE predictive, not reactive)
## Estimated complexity: High

---

## What This Is

Implementation of the **Free Energy Principle** for ORE. The system maintains predictions about incoming signals and learns from prediction errors. This transforms ORE from a reactive system to a **predictive** one.

- **Predict** what the next input will be
- **Compare** prediction to actual input
- **Update** model based on prediction error
- **Act** to minimize future prediction error

---

## Why It Matters

**A5 (Continual Learning):** "Prediction error is the universal learning signal. It's how brains learn, how attention is allocated, how surprise is computed. Without it, ORE just reacts."

**P1 (Dynamical Systems):** "Free energy minimization gives the system a *goal* - reduce uncertainty. This is intrinsic motivation without hardcoded rewards."

**H3 (Enactivism):** "Active inference closes the perception-action loop. The system doesn't just model the world - it acts to make the world match its model."

---

## The Core Insight

The system maintains **generative models** that predict:
1. What phase patterns will occur next (temporal prediction)
2. What input will arrive (sensory prediction)
3. What the body state will be (interoceptive prediction)

**Prediction error** = actual - predicted

High prediction error → **surprise** → triggers:
- Learning (update model)
- Attention (focus on surprising input)
- Action (change world to match prediction)

```
      ┌─────────────────┐
      │ Generative Model │
      │  (predictions)   │
      └────────┬────────┘
               │ predict
               ▼
      ┌─────────────────┐
      │   Comparator    │◄──── actual input
      │                 │
      └────────┬────────┘
               │ prediction error
               ▼
    ┌──────────┴──────────┐
    ▼                     ▼
┌────────┐          ┌──────────┐
│ Update │          │  Action  │
│ Model  │          │Selection │
└────────┘          └──────────┘
```

---

## Interface Contract

```python
class ActiveInferenceEngine:
    """
    Active inference for ORE substrate.
    
    Properties:
        free_energy: float              # Current free energy (lower = better)
        prediction_error: float         # Current prediction error magnitude
        surprise: float                 # -log p(observation)
        expected_free_energy: Dict[str, float]  # Per-action EFE
    
    Methods:
        # Prediction
        predict_next_state(current_state) -> PredictedState
        compute_prediction_error(predicted, actual) -> PredictionError
        
        # Learning
        update_model(prediction_error)
        
        # Action selection
        compute_expected_free_energy(actions) -> Dict[str, float]
        select_action(actions) -> str
        
        # Integration
        step(observation, available_actions) -> InferenceResult
"""

@dataclass
class PredictedState:
    """Predicted future state."""
    fast_phases: np.ndarray          # Predicted fast oscillator phases
    slow_phases: np.ndarray          # Predicted slow oscillator phases
    coherence: float                 # Predicted coherence
    valence: float                   # Predicted valence
    confidence: float                # How confident in prediction (0-1)

@dataclass
class PredictionError:
    """Decomposed prediction error."""
    phase_error: float               # Error in phase predictions
    coherence_error: float           # Error in coherence prediction
    valence_error: float             # Error in valence prediction
    total: float                     # Combined error
    surprise: float                  # Information-theoretic surprise

@dataclass
class InferenceResult:
    """Result of one inference step."""
    prediction: PredictedState
    actual: dict
    error: PredictionError
    free_energy: float
    selected_action: Optional[str]
    model_updated: bool
```

---

## Configuration

```python
@dataclass
class ActiveInferenceConfig:
    # Prediction
    prediction_horizon: int = 5        # Steps ahead to predict
    phase_prediction_weight: float = 0.4
    coherence_prediction_weight: float = 0.3
    valence_prediction_weight: float = 0.3
    
    # Learning
    learning_rate: float = 0.01
    precision_learning_rate: float = 0.005  # For uncertainty estimation
    min_precision: float = 0.1
    max_precision: float = 10.0
    
    # Action selection
    action_precision: float = 4.0      # Inverse temperature for softmax
    epistemic_value_weight: float = 0.5  # Curiosity
    pragmatic_value_weight: float = 0.5  # Goal-seeking
    
    # Free energy
    complexity_penalty: float = 0.1    # Penalize complex models
```

---

## Generative Model

The generative model predicts future states given current state:

```python
class GenerativeModel:
    """
    Learned model that generates predictions.
    
    Uses linear dynamics in phase space with learned transition matrices.
    """
    
    def __init__(self, fast_dim: int, slow_dim: int, config: ActiveInferenceConfig):
        self.config = config
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        
        # Transition matrices (learned)
        # Predict next phases from current phases
        self.A_fast = np.eye(fast_dim) + 0.01 * np.random.randn(fast_dim, fast_dim)
        self.A_slow = np.eye(slow_dim) + 0.01 * np.random.randn(slow_dim, slow_dim)
        
        # Cross-scale prediction
        self.B_fast_to_slow = np.zeros((slow_dim, fast_dim))
        self.B_slow_to_fast = np.zeros((fast_dim, slow_dim))
        
        # Precision (inverse variance) - learned confidence
        self.precision_fast = np.ones(fast_dim)
        self.precision_slow = np.ones(slow_dim)
        self.precision_coherence = 1.0
        self.precision_valence = 1.0
        
        # History for learning
        self._state_history: List[dict] = []
    
    def predict(self, current_state: dict) -> PredictedState:
        """Generate prediction for next state."""
        fast_phases = np.array(current_state['fast_phases'])
        slow_phases = np.array(current_state['slow_phases'])
        
        # Linear prediction with phase wrapping
        pred_fast = (self.A_fast @ fast_phases + 
                     self.B_slow_to_fast @ slow_phases) % (2 * np.pi)
        pred_slow = (self.A_slow @ slow_phases + 
                     self.B_fast_to_slow @ fast_phases) % (2 * np.pi)
        
        # Predict coherence from phases
        pred_coherence = self._predict_coherence(pred_fast, pred_slow)
        
        # Predict valence (simple autoregressive)
        pred_valence = current_state.get('valence', 0) * 0.9
        
        # Confidence from precision
        confidence = np.mean([
            np.mean(self.precision_fast),
            np.mean(self.precision_slow),
            self.precision_coherence,
        ]) / self.config.max_precision
        
        return PredictedState(
            fast_phases=pred_fast,
            slow_phases=pred_slow,
            coherence=pred_coherence,
            valence=pred_valence,
            confidence=np.clip(confidence, 0, 1),
        )
    
    def _predict_coherence(self, fast_phases: np.ndarray, slow_phases: np.ndarray) -> float:
        """Predict coherence from phase pattern."""
        combined = np.concatenate([fast_phases, slow_phases])
        return min(np.abs(np.mean(np.exp(1j * combined))), 0.999)
    
    def update(self, prediction: PredictedState, actual: dict, error: 'PredictionError'):
        """Update model based on prediction error."""
        cfg = self.config
        lr = cfg.learning_rate
        
        fast_actual = np.array(actual['fast_phases'])
        slow_actual = np.array(actual['slow_phases'])
        
        # Compute phase gradients (circular difference)
        fast_diff = np.angle(np.exp(1j * (fast_actual - prediction.fast_phases)))
        slow_diff = np.angle(np.exp(1j * (slow_actual - prediction.slow_phases)))
        
        # Update transition matrices toward reducing error
        # Gradient descent on prediction error
        fast_outer = np.outer(fast_diff, np.array(actual['fast_phases']))
        slow_outer = np.outer(slow_diff, np.array(actual['slow_phases']))
        
        self.A_fast += lr * fast_outer / self.fast_dim
        self.A_slow += lr * slow_outer / self.slow_dim
        
        # Update precision (certainty) based on error magnitude
        # Low error → increase precision, high error → decrease
        precision_lr = cfg.precision_learning_rate
        
        fast_error_per_osc = np.abs(fast_diff)
        slow_error_per_osc = np.abs(slow_diff)
        
        self.precision_fast += precision_lr * (1 / (fast_error_per_osc + 0.1) - self.precision_fast)
        self.precision_slow += precision_lr * (1 / (slow_error_per_osc + 0.1) - self.precision_slow)
        
        # Clamp precision
        self.precision_fast = np.clip(self.precision_fast, cfg.min_precision, cfg.max_precision)
        self.precision_slow = np.clip(self.precision_slow, cfg.min_precision, cfg.max_precision)
```

---

## Method Specifications

### `predict_next_state(current_state) -> PredictedState`

```python
def predict_next_state(self, current_state: dict) -> PredictedState:
    """
    Generate prediction for next state.
    
    current_state should contain:
        - fast_phases: [fast_dim] array
        - slow_phases: [slow_dim] array
        - coherence: float
        - valence: float
    """
    return self.model.predict(current_state)
```

### `compute_prediction_error(predicted, actual) -> PredictionError`

```python
def compute_prediction_error(self, 
                             predicted: PredictedState, 
                             actual: dict) -> PredictionError:
    """
    Compute prediction error between predicted and actual state.
    """
    cfg = self.config
    
    # Phase error (circular distance)
    fast_actual = np.array(actual['fast_phases'])
    slow_actual = np.array(actual['slow_phases'])
    
    fast_diff = np.angle(np.exp(1j * (fast_actual - predicted.fast_phases)))
    slow_diff = np.angle(np.exp(1j * (slow_actual - predicted.slow_phases)))
    
    # Weighted by precision (certain predictions count more)
    phase_error_fast = np.mean(np.abs(fast_diff) * self.model.precision_fast)
    phase_error_slow = np.mean(np.abs(slow_diff) * self.model.precision_slow)
    phase_error = (phase_error_fast + phase_error_slow) / 2
    
    # Coherence error
    coherence_actual = actual.get('coherence', 0)
    coherence_error = abs(coherence_actual - predicted.coherence) * self.model.precision_coherence
    
    # Valence error
    valence_actual = actual.get('valence', 0)
    valence_error = abs(valence_actual - predicted.valence) * self.model.precision_valence
    
    # Total weighted error
    total = (
        cfg.phase_prediction_weight * phase_error +
        cfg.coherence_prediction_weight * coherence_error +
        cfg.valence_prediction_weight * valence_error
    )
    
    # Surprise = -log p(observation)
    # Approximated as proportional to precision-weighted error
    surprise = total * (1 + predicted.confidence)
    
    return PredictionError(
        phase_error=phase_error,
        coherence_error=coherence_error,
        valence_error=valence_error,
        total=total,
        surprise=surprise,
    )
```

### `update_model(prediction_error)`

```python
def update_model(self, 
                 prediction: PredictedState,
                 actual: dict,
                 error: PredictionError):
    """
    Update generative model based on prediction error.
    """
    self.model.update(prediction, actual, error)
    
    # Track free energy history
    self._free_energy_history.append(self.free_energy)
```

### `compute_expected_free_energy(actions) -> Dict[str, float]`

Expected Free Energy (EFE) for action selection.

```python
def compute_expected_free_energy(self, 
                                  current_state: dict,
                                  actions: List[str]) -> Dict[str, float]:
    """
    Compute expected free energy for each possible action.
    
    EFE = pragmatic_value + epistemic_value
    
    Pragmatic: Does action lead to preferred states?
    Epistemic: Does action reduce uncertainty?
    
    Lower EFE = better action.
    """
    cfg = self.config
    efe = {}
    
    for action in actions:
        # Simulate action effect on state
        simulated_state = self._simulate_action(current_state, action)
        
        # Predict what would happen after action
        predicted_next = self.model.predict(simulated_state)
        
        # Pragmatic value: distance from preferred state
        # Preferred = high coherence, neutral valence
        preferred_coherence = 0.7
        preferred_valence = 0.0
        
        pragmatic = (
            abs(predicted_next.coherence - preferred_coherence) +
            abs(predicted_next.valence - preferred_valence)
        )
        
        # Epistemic value: expected information gain
        # High uncertainty states → exploring them reduces uncertainty
        epistemic = -np.mean(1 / (self.model.precision_fast + 0.1))
        epistemic += -np.mean(1 / (self.model.precision_slow + 0.1))
        
        # For actions that lead to uncertain areas, epistemic is negative (good)
        # We want to explore to reduce uncertainty
        if predicted_next.confidence < 0.5:
            epistemic -= 0.2  # Bonus for exploring uncertain territory
        
        # Combined EFE (lower is better)
        efe[action] = (
            cfg.pragmatic_value_weight * pragmatic +
            cfg.epistemic_value_weight * epistemic
        )
    
    return efe

def _simulate_action(self, current_state: dict, action: str) -> dict:
    """Simulate effect of action on state."""
    # Simplified simulation - actions affect phases
    simulated = current_state.copy()
    
    action_effects = {
        'focus': {'coherence_boost': 0.1},
        'explore': {'phase_noise': 0.2},
        'rest': {'valence_boost': 0.05},
        'engage': {'activation_boost': 0.1},
        'withdraw': {'activation_decay': 0.2},
    }
    
    effect = action_effects.get(action, {})
    
    if 'coherence_boost' in effect:
        # Move phases toward mean
        mean_phase = np.mean(simulated['slow_phases'])
        simulated['slow_phases'] = (
            0.9 * simulated['slow_phases'] + 0.1 * mean_phase
        )
    
    if 'phase_noise' in effect:
        simulated['fast_phases'] = (
            simulated['fast_phases'] + 
            effect['phase_noise'] * np.random.randn(len(simulated['fast_phases']))
        ) % (2 * np.pi)
    
    return simulated
```

### `select_action(actions) -> str`

```python
def select_action(self, 
                  current_state: dict,
                  actions: List[str]) -> str:
    """
    Select action with lowest expected free energy.
    
    Uses softmax with action_precision as inverse temperature.
    """
    cfg = self.config
    
    efe = self.compute_expected_free_energy(current_state, actions)
    
    # Softmax selection (lower EFE = higher probability)
    efe_values = np.array([efe[a] for a in actions])
    probs = np.exp(-cfg.action_precision * efe_values)
    probs = probs / (probs.sum() + 1e-8)
    
    # Sample action
    selected_idx = np.random.choice(len(actions), p=probs)
    return actions[selected_idx]
```

### `step(observation, available_actions) -> InferenceResult`

Main inference loop.

```python
def step(self, 
         observation: dict,
         available_actions: List[str] = None) -> InferenceResult:
    """
    One step of active inference.
    
    1. Predict what we expected
    2. Compare to observation
    3. Update model
    4. Select action (if actions available)
    """
    # Get prediction (made at end of last step)
    if self._pending_prediction is None:
        prediction = self.model.predict(observation)
    else:
        prediction = self._pending_prediction
    
    # Compute prediction error
    error = self.compute_prediction_error(prediction, observation)
    
    # Update model
    self.update_model(prediction, observation, error)
    self._model_updated = True
    
    # Compute free energy
    # F = complexity + inaccuracy
    complexity = self.config.complexity_penalty * self._model_complexity()
    inaccuracy = error.total
    self._free_energy = complexity + inaccuracy
    
    # Select action if available
    selected_action = None
    if available_actions:
        selected_action = self.select_action(observation, available_actions)
    
    # Make prediction for next step
    self._pending_prediction = self.model.predict(observation)
    
    return InferenceResult(
        prediction=prediction,
        actual=observation,
        error=error,
        free_energy=self._free_energy,
        selected_action=selected_action,
        model_updated=self._model_updated,
    )

def _model_complexity(self) -> float:
    """Compute model complexity (for free energy)."""
    # Simplified: count non-near-identity weights
    identity_fast = np.eye(self.model.fast_dim)
    identity_slow = np.eye(self.model.slow_dim)
    
    deviation_fast = np.sum(np.abs(self.model.A_fast - identity_fast))
    deviation_slow = np.sum(np.abs(self.model.A_slow - identity_slow))
    
    return (deviation_fast + deviation_slow) / (self.model.fast_dim + self.model.slow_dim)
```

---

## Properties

```python
@property
def free_energy(self) -> float:
    """Current variational free energy."""
    return self._free_energy

@property
def prediction_error(self) -> float:
    """Most recent prediction error magnitude."""
    return self._last_error.total if self._last_error else 0.0

@property
def surprise(self) -> float:
    """Most recent surprise value."""
    return self._last_error.surprise if self._last_error else 0.0

@property
def model_confidence(self) -> float:
    """Overall model confidence (from precision)."""
    return np.mean([
        np.mean(self.model.precision_fast),
        np.mean(self.model.precision_slow),
    ]) / self.config.max_precision
```

---

## Integration with Entity

```python
# In DevelopmentalEntity.__init__:
self.inference = ActiveInferenceEngine(
    fast_dim=self.substrate.fast.n,
    slow_dim=self.substrate.slow.n,
)

# In DevelopmentalEntity.tick():
def tick(self):
    # ... existing tick code ...
    
    # Active inference step
    observation = {
        'fast_phases': self.substrate.fast.phases.copy(),
        'slow_phases': self.substrate.slow.phases.copy(),
        'coherence': self.substrate.global_coherence,
        'valence': self.body.valence,
    }
    
    inference_result = self.inference.step(
        observation,
        available_actions=['focus', 'explore', 'rest', 'engage', 'withdraw']
    )
    
    # Apply selected action
    if inference_result.selected_action:
        self._apply_action(inference_result.selected_action)
    
    # Use surprise to modulate arousal
    self.body.arousal += 0.1 * inference_result.error.surprise
    self.body.arousal = np.clip(self.body.arousal, 0, 1)
    
    return {**result, 'inference': inference_result}
```

---

## Success Criteria

### Prediction
1. Prediction error decreases over time on consistent input
2. Surprise spikes on novel input
3. Confidence increases as model learns patterns

### Learning
1. Transition matrices adapt to actual dynamics
2. Precision tracks prediction reliability per oscillator
3. Model complexity stays bounded

### Action Selection
1. Actions with lower EFE selected more often
2. System balances exploration (epistemic) and exploitation (pragmatic)
3. Actions affect state as expected

---

## Test Cases

```python
def test_prediction_improves():
    """Prediction error should decrease on repeated patterns."""
    engine = ActiveInferenceEngine(fast_dim=20, slow_dim=10)
    
    # Generate consistent oscillation pattern
    errors = []
    for i in range(100):
        state = {
            'fast_phases': (np.arange(20) * 0.1 + i * 0.05) % (2 * np.pi),
            'slow_phases': (np.arange(10) * 0.2 + i * 0.02) % (2 * np.pi),
            'coherence': 0.6,
            'valence': -0.1,
        }
        result = engine.step(state)
        errors.append(result.error.total)
    
    # Error should decrease
    assert np.mean(errors[-10:]) < np.mean(errors[:10])

def test_surprise_on_novelty():
    """Surprise should spike on unexpected input."""
    engine = ActiveInferenceEngine(fast_dim=20, slow_dim=10)
    
    # Train on consistent pattern
    for i in range(50):
        state = {
            'fast_phases': np.zeros(20),
            'slow_phases': np.zeros(10),
            'coherence': 0.5,
            'valence': 0.0,
        }
        engine.step(state)
    
    baseline_surprise = engine.surprise
    
    # Novel input
    novel_state = {
        'fast_phases': np.random.uniform(0, 2*np.pi, 20),
        'slow_phases': np.random.uniform(0, 2*np.pi, 10),
        'coherence': 0.9,
        'valence': -0.5,
    }
    result = engine.step(novel_state)
    
    assert result.error.surprise > baseline_surprise * 2

def test_action_selection():
    """Should select actions based on EFE."""
    engine = ActiveInferenceEngine(fast_dim=20, slow_dim=10)
    
    state = {
        'fast_phases': np.random.uniform(0, 2*np.pi, 20),
        'slow_phases': np.random.uniform(0, 2*np.pi, 10),
        'coherence': 0.3,  # Low coherence
        'valence': -0.3,   # Negative valence
    }
    
    # With low coherence, 'focus' should often be selected
    actions = ['focus', 'explore', 'rest', 'engage', 'withdraw']
    selected = [engine.select_action(state, actions) for _ in range(100)]
    
    # Focus should be common (moves toward preferred high-coherence state)
    focus_count = selected.count('focus')
    assert focus_count > 10  # At least sometimes selected
```

---

## Dependencies

- `numpy`
- `MultiScaleSubstrate` (ORE2-002)
- `EmbodimentLayer` (ORE2-003)

---

## File Location

```
ore2/
├── core/
│   └── active_inference.py  # <-- This component
├── tests/
│   └── test_active_inference.py
```

---

## Design Decisions to Preserve

1. **Linear dynamics in phase space** - Simple but learnable transition model
2. **Precision-weighted errors** - Confident predictions count more
3. **Circular phase distance** - Proper handling of angular variables
4. **Epistemic + pragmatic value** - Balances exploration and exploitation
5. **Surprise modulates arousal** - Unexpected input increases alertness
6. **Complexity penalty** - Prevents overfitting
