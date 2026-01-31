# BRIEFING: Dashboard

## Component ID: ORE2-013
## Priority: Medium (Observability)
## Estimated complexity: Medium

---

## What This Is

Real-time visualization of ORE entity state. A web-based dashboard that shows:
- Oscillator phases and coherence (animated)
- Memory tree growth
- CI over time
- Claim activations
- Development stage progress
- Multi-agent network graph (if applicable)

This is how you **see** what ORE is doing, not just read numbers.

---

## Why It Matters

**H2 (Philosophy of Mind):** "Without visibility, it's a black box. You need to see the dynamics to understand them. Coherence isn't just a number - it's synchronization you can watch happen."

**I1 (Systems Architect):** "Debugging distributed dynamics is hard. A real-time view of phase space is worth a thousand print statements."

**X6 (Enterprise):** "Stakeholders want to see something. A dashboard showing 'consciousness index rising' is more compelling than JSON."

---

## The Core Insight

The dashboard connects to EntityAPI's WebSocket and visualizes the streams:

```
┌─────────────────────────────────────────────────────────────────┐
│                     ORE Dashboard                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  Phase Circle    │  │  CI Timeline     │  │  Memory Tree │  │
│  │  (animated)      │  │                  │  │              │  │
│  │    ◐◐◐◑◑         │  │  ╱╲  ╱╲         │  │    ●         │  │
│  │   ◐    ◑        │  │ ╱  ╲╱  ╲        │  │   /│\        │  │
│  │  ◐      ◑       │  │╱       ╲       │  │  ● ● ●      │  │
│  │   ◐    ◑        │  │         ╲      │  │              │  │
│  │    ◐◐◐◑◑         │  │                  │  │              │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  Body State      │  │  Active Claims   │  │  Dev Stage   │  │
│  │                  │  │                  │  │              │  │
│  │  ♥ 72 bpm       │  │  • Analyst mode  │  │  [████░░░░]  │  │
│  │  Energy: ███░░  │  │  • Value clarity │  │  BABBLING    │  │
│  │  Valence: -0.12 │  │                  │  │  34%         │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Interface Contract

```python
class Dashboard:
    """
    Web-based visualization dashboard.
    
    Methods:
        # Server
        start(host, port)
        stop()
        
        # Connection
        connect_to_entity(api_url, entity_id, api_key)
        disconnect()
        
        # Configuration
        set_update_rate(hz)
        toggle_component(component_name, visible)
"""

# Dashboard doesn't need much Python - it's mostly a frontend
# The Python component just serves static files and proxies WebSocket
```

---

## Components

### 1. Phase Circle (Real-time)

Animated visualization of oscillator phases as points on a circle.

```
Fast Scale                    Slow Scale
    ◐                            ◐
  ◐   ◐                        ◐   ◐
 ◐     ◐   coherence=0.7      ◐     ◐   coherence=0.8
  ◐   ◐                        ◐   ◐
    ◐                            ◐
```

- Each dot is an oscillator
- Position on circle = phase angle
- Color = activation potential (dim = dormant, bright = active)
- Clustering = high coherence
- Spread = low coherence

**Data source:** `coherence` WebSocket channel

### 2. CI Timeline

Line chart of Consciousness Index over time.

```
CI
1.0│
   │     ╱╲
0.5│   ╱    ╲    ╱
   │ ╱        ╲╱
0.0│────────────────────
   0     time     now
```

- Show CI_integrated, CI_fast, CI_slow as separate lines
- Highlight attractor entry/exit
- Mark significant events (experiences, rest periods)

**Data source:** `cognitive_state` WebSocket channel

### 3. Memory Tree

Hierarchical visualization of merkle memory.

```
        [ROOT]
       /  |  \  \
    SELF REL INS EXP
     |    |   |   |
     ●    ●   ●   ●●●●
           |
           ●
```

- Four branches visible
- Nodes sized by coherence at creation
- Grain boundaries shown as red edges
- Click node to see content

**Data source:** `memory_events` WebSocket channel + initial state fetch

### 4. Body State

Gauges and indicators for embodiment.

```
♥ Heartbeat: ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁ (72 bpm)

Energy:  [████████░░] 0.82
Arousal: [████░░░░░░] 0.41
Valence: [░░░░██░░░░] -0.12
              ↑ baseline
```

- Heartbeat as animated waveform
- Energy/arousal as bar gauges
- Valence as centered bar (negative left, positive right)

**Data source:** `cognitive_state` WebSocket channel

### 5. Active Claims

List of currently active claims with strength.

```
Active Claims (4)
─────────────────────────────
■■■■■■■■░░ I examine problems systematically
■■■■■■░░░░ I value clear communication
■■■■░░░░░░ I seek evidence
■■■░░░░░░░ I question assumptions
```

- Sorted by strength
- Color by scope (identity=blue, behavior=green, etc.)
- Click to see full content and source

**Data source:** `claim_activations` WebSocket channel

### 6. Development Stage

Progress through developmental stages.

```
┌─────────────────────────────────────────┐
│ GENESIS ✓ │ BABBLING ▶ │ IMITATION │ ... │
└─────────────────────────────────────────┘

Stage: BABBLING (34%)
Age: 1,523 ticks
Oscillators: 45/200
Experiences: 47 (12 significant)
Milestones: 3
```

- Stage progression as tabs
- Current stage highlighted
- Progress bar within stage

**Data source:** Initial state + updates from `cognitive_state`

### 7. Network Graph (Multi-agent)

Force-directed graph of entity network.

```
     (Alice)
      /   \
     /     \
(Bob)──────(Carol)
     \     /
      \   /
     (Dave)

Edge thickness = coupling strength
Edge color = trust level
Node size = CI
Node pulse = coherence
```

**Data source:** Network state endpoint

---

## Frontend Stack

```
React + D3.js
├── components/
│   ├── PhaseCircle.tsx      # Animated SVG
│   ├── CITimeline.tsx       # D3 line chart
│   ├── MemoryTree.tsx       # D3 tree layout
│   ├── BodyState.tsx        # Gauges
│   ├── ClaimsList.tsx       # List component
│   ├── DevStage.tsx         # Progress display
│   └── NetworkGraph.tsx     # Force-directed graph
├── hooks/
│   ├── useWebSocket.ts      # WebSocket connection
│   └── useEntityState.ts    # State management
├── App.tsx
└── index.tsx
```

---

## Key Visualizations

### Phase Circle Animation

```typescript
// PhaseCircle.tsx
function PhaseCircle({ phases, activations, coherence }) {
  const svgRef = useRef();
  
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const width = 200, height = 200;
    const radius = 80;
    const center = { x: width/2, y: height/2 };
    
    // Update oscillator positions
    const dots = svg.selectAll('.oscillator')
      .data(phases);
    
    dots.enter()
      .append('circle')
      .attr('class', 'oscillator')
      .attr('r', 4)
      .merge(dots)
      .transition()
      .duration(50)  // Smooth animation
      .attr('cx', (phase, i) => center.x + radius * Math.cos(phase))
      .attr('cy', (phase, i) => center.y + radius * Math.sin(phase))
      .attr('fill', (_, i) => `rgba(66, 135, 245, ${activations[i]})`);
    
    // Update coherence display
    svg.select('.coherence-text')
      .text(`C = ${coherence.toFixed(3)}`);
      
  }, [phases, activations, coherence]);
  
  return <svg ref={svgRef} width={200} height={200} />;
}
```

### CI Timeline

```typescript
// CITimeline.tsx
function CITimeline({ history }) {
  const svgRef = useRef();
  
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const width = 400, height = 150;
    const margin = { top: 10, right: 10, bottom: 20, left: 40 };
    
    const x = d3.scaleTime()
      .domain(d3.extent(history, d => d.timestamp))
      .range([margin.left, width - margin.right]);
    
    const y = d3.scaleLinear()
      .domain([0, 1])
      .range([height - margin.bottom, margin.top]);
    
    const line = d3.line()
      .x(d => x(d.timestamp))
      .y(d => y(d.ci))
      .curve(d3.curveMonotoneX);
    
    svg.select('.ci-line')
      .datum(history)
      .attr('d', line);
      
  }, [history]);
  
  return <svg ref={svgRef} width={400} height={150} />;
}
```

### Memory Tree

```typescript
// MemoryTree.tsx
function MemoryTree({ nodes, branches }) {
  const svgRef = useRef();
  
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    
    // Build hierarchy from nodes
    const root = d3.hierarchy(buildTree(nodes, branches));
    
    const treeLayout = d3.tree()
      .size([360, 100])
      .separation((a, b) => (a.parent === b.parent ? 1 : 2));
    
    treeLayout(root);
    
    // Draw links
    svg.selectAll('.link')
      .data(root.links())
      .join('path')
      .attr('class', 'link')
      .attr('d', d3.linkRadial()
        .angle(d => d.x * Math.PI / 180)
        .radius(d => d.y));
    
    // Draw nodes
    svg.selectAll('.node')
      .data(root.descendants())
      .join('circle')
      .attr('class', 'node')
      .attr('r', d => 4 + d.data.coherence * 6)
      .attr('transform', d => `rotate(${d.x - 90}) translate(${d.y},0)`);
      
  }, [nodes, branches]);
  
  return <svg ref={svgRef} width={300} height={300} />;
}
```

---

## WebSocket Integration

```typescript
// useWebSocket.ts
function useEntityWebSocket(apiUrl: string, entityId: string, apiKey: string) {
  const [state, setState] = useState<EntityState | null>(null);
  const [coherenceHistory, setCoherenceHistory] = useState<CoherencePoint[]>([]);
  
  useEffect(() => {
    const ws = new WebSocket(`${apiUrl}/ws/${entityId}?api_key=${apiKey}`);
    
    ws.onopen = () => {
      ws.send(JSON.stringify({
        type: 'subscribe',
        channels: ['cognitive_state', 'coherence', 'memory_events', 'claim_activations']
      }));
    };
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      switch (message.type) {
        case 'cognitive_state':
          setState(message.data);
          break;
          
        case 'coherence':
          setCoherenceHistory(prev => [...prev.slice(-500), {
            timestamp: new Date(message.timestamp),
            ...message.data
          }]);
          break;
          
        case 'memory_event':
          // Update memory tree
          break;
          
        case 'claim_activation':
          // Update claims list
          break;
      }
    };
    
    return () => ws.close();
  }, [apiUrl, entityId, apiKey]);
  
  return { state, coherenceHistory };
}
```

---

## Python Server (Serves Dashboard)

```python
class DashboardServer:
    """Serves the dashboard static files and proxies WebSocket."""
    
    def __init__(self, 
                 static_dir: str = "./dashboard/build",
                 api_url: str = "http://localhost:8000"):
        self.app = FastAPI()
        self.static_dir = static_dir
        self.api_url = api_url
        
        # Serve static files
        self.app.mount("/", StaticFiles(directory=static_dir, html=True))
        
        # Config endpoint
        self.app.get("/config")(self.get_config)
    
    def get_config(self):
        """Return API URL for frontend."""
        return {"api_url": self.api_url}
    
    def start(self, host: str = "0.0.0.0", port: int = 3000):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)
```

---

## Success Criteria

### Visualization
1. Phase circle animates smoothly at 10+ FPS
2. CI timeline updates in real-time
3. Memory tree grows as nodes are added
4. All components reflect actual state

### Usability
1. Dashboard loads in <2 seconds
2. Responsive layout (works on different screen sizes)
3. Components can be toggled on/off
4. Dark/light theme support

### Performance
1. Handles 100+ oscillators without lag
2. Memory tree handles 1000+ nodes
3. WebSocket reconnects on disconnect

---

## File Structure

```
dashboard/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── PhaseCircle.tsx
│   │   ├── CITimeline.tsx
│   │   ├── MemoryTree.tsx
│   │   ├── BodyState.tsx
│   │   ├── ClaimsList.tsx
│   │   ├── DevStage.tsx
│   │   └── NetworkGraph.tsx
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   └── useEntityState.ts
│   ├── App.tsx
│   ├── App.css
│   └── index.tsx
├── package.json
└── tsconfig.json

ore2/
├── dashboard/
│   └── server.py  # Python server to serve built dashboard
```

---

## Design Decisions to Preserve

1. **WebSocket for real-time** - Polling is too slow for phase animation
2. **D3 for complex viz** - Phase circles and trees need SVG
3. **React for state** - Component model fits dashboard well
4. **Separate from API** - Dashboard is optional, doesn't affect core
5. **10 FPS minimum** - Below this, animation looks choppy
6. **500-point history** - Enough for trends, not too much memory

---

## Future Enhancements (Not MVP)

- 3D phase visualization (Three.js)
- Sound synthesis from coherence
- VR view of entity network
- Recording/playback of sessions
- Export visualizations as video
