import cytoscape, { type ElementDefinition } from "cytoscape";
import {
  startTransition,
  type FormEvent,
  useDeferredValue,
  useEffect,
  useRef,
  useState
} from "react";
import { fetchAnswer, fetchGraphNeighborhood, fetchSearch, getHealth } from "./api";
import type {
  AnswerMode,
  AnswerResponse,
  GraphEdgePayload,
  GraphNeighborhoodResponse,
  GraphNodePayload,
  SearchResponse
} from "./types";

const EXAMPLE_QUERIES = [
  "What does the architecture say about the execution layer?",
  "Show me clusters around RAG",
  "Which notes connect retrieval and automation?"
] as const;

const ANSWER_MODES: AnswerMode[] = [
  "normal",
  "technical",
  "concise",
  "detailed",
  "bullet_summary"
];

export default function App() {
  const [query, setQuery] = useState<string>(EXAMPLE_QUERIES[0]);
  const deferredQuery = useDeferredValue(query);
  const [answerMode, setAnswerMode] = useState<AnswerMode>("normal");
  const [graph, setGraph] = useState<GraphNeighborhoodResponse | null>(null);
  const [answer, setAnswer] = useState<AnswerResponse | null>(null);
  const [search, setSearch] = useState<SearchResponse | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [statusLabel, setStatusLabel] = useState("Checking backend");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [lastSubmittedQuery, setLastSubmittedQuery] = useState<string>("");

  const graphContainerRef = useRef<HTMLDivElement | null>(null);
  const cytoscapeRef = useRef<cytoscape.Core | null>(null);

  useEffect(() => {
    let canceled = false;
    getHealth()
      .then((payload) => {
        if (canceled) {
          return;
        }
        setStatusLabel(`${payload.service} backend ready`);
      })
      .catch((error: Error) => {
        if (canceled) {
          return;
        }
        setStatusLabel("Backend unavailable");
        setErrorMessage(error.message);
      });

    return () => {
      canceled = true;
    };
  }, []);

  useEffect(() => {
    if (!graphContainerRef.current || cytoscapeRef.current) {
      return;
    }

    const cy = cytoscape({
      container: graphContainerRef.current,
      elements: [],
      layout: { name: "preset" },
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "text-wrap": "wrap",
            "text-max-width": 140,
            "font-size": 11,
            "font-family": "Space Grotesk, Avenir Next, Segoe UI, sans-serif",
            color: "#f4f0e8",
            "text-valign": "center",
            "text-halign": "center",
            "border-width": 1,
            "border-color": "#eec57c",
            "background-opacity": "0.95"
          }
        },
        {
          selector: 'node[type = "document"]',
          style: {
            shape: "round-rectangle",
            width: 180,
            height: 70,
            "background-color": "#16343f"
          }
        },
        {
          selector: 'node[type = "chunk"]',
          style: {
            shape: "ellipse",
            width: 100,
            height: 100,
            "background-color": "#845ef7"
          }
        },
        {
          selector: "node[highlighted = 1]",
          style: {
            "border-width": 3,
            "border-color": "#ffe7a8"
          }
        },
        {
          selector: "node[selected = 1]",
          style: {
            "overlay-color": "#ffffff",
            "overlay-opacity": 0.12,
            "border-color": "#ffffff"
          }
        },
        {
          selector: "edge",
          style: {
            width: 2,
            opacity: 0.66,
            "curve-style": "bezier"
          }
        },
        {
          selector: 'edge[type = "belongs_to"]',
          style: {
            "line-color": "#4db5ae"
          }
        },
        {
          selector: 'edge[type = "similar_to"]',
          style: {
            "line-color": "#f08d49",
            "line-style": "dashed"
          }
        }
      ] as never
    });

    cy.on("tap", "node", (event) => {
      const nodeId = event.target.id();
      setSelectedNodeId(nodeId);
    });

    cytoscapeRef.current = cy;

    return () => {
      cy.destroy();
      cytoscapeRef.current = null;
    };
  }, []);

  useEffect(() => {
    const cy = cytoscapeRef.current;
    if (!cy) {
      return;
    }

    const elements = graph ? buildElements(graph, selectedNodeId) : [];
    cy.elements().remove();
    cy.add(elements);

    if (elements.length > 0) {
      cy.layout({
        name: "cose",
        animate: false,
        fit: true,
        padding: 36,
        nodeRepulsion: 110000,
        idealEdgeLength: 120
      }).run();
    }
  }, [graph, selectedNodeId]);

  async function handleSubmit(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();

    const nextQuery = query.trim();
    if (!nextQuery) {
      setErrorMessage("Enter a query before asking CortexRAG to light up the graph.");
      return;
    }

    setIsSubmitting(true);
    setErrorMessage(null);

    try {
      const [graphPayload, answerPayload, searchPayload] = await Promise.all([
        fetchGraphNeighborhood(nextQuery),
        fetchAnswer(nextQuery, answerMode),
        fetchSearch(nextQuery)
      ]);

      startTransition(() => {
        setGraph(graphPayload);
        setAnswer(answerPayload);
        setSearch(searchPayload);
        setLastSubmittedQuery(nextQuery);
        setSelectedNodeId(graphPayload.seed_node_ids[0] ?? graphPayload.nodes[0]?.id ?? null);
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown UI request failure.";
      setErrorMessage(message);
    } finally {
      setIsSubmitting(false);
    }
  }

  const selectedNode = graph?.nodes.find((node) => node.id === selectedNodeId) ?? null;
  const selectedEdges = graph ? relatedEdges(graph.edges, selectedNodeId) : [];
  const connectedNodes = graph ? relatedNodes(graph.nodes, selectedEdges, selectedNodeId) : [];

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <header className="hero-bar">
        <div>
          <p className="eyebrow">Local Knowledge Operating System</p>
          <h1>CortexRAG Brain View</h1>
          <p className="hero-copy">
            Query your local graph, inspect the retrieval neighborhood, and read the grounded answer
            without leaving the visual context.
          </p>
        </div>

        <div className="hero-metrics">
          <MetricCard label="Backend" value={statusLabel} tone={errorMessage ? "warn" : "ok"} />
          <MetricCard label="Graph" value={graph ? `${graph.nodes.length} nodes` : "Awaiting query"} />
          <MetricCard label="Answer" value={answer?.generated ? answer.model ?? "Generated" : "No answer yet"} />
        </div>
      </header>

      <main className="workspace-grid">
        <section className="panel panel-canvas">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Graph Mode</p>
              <h2>Knowledge canvas</h2>
            </div>
            <div className="panel-badges">
              <span className="badge">Documents + Chunks</span>
              <span className="badge">Belongs + Similarity</span>
            </div>
          </div>

          <div className="canvas-stage">
            <div ref={graphContainerRef} className="graph-canvas" />
            {!graph && (
              <div className="canvas-empty">
                <p>Run a query to render the first graph neighborhood.</p>
                <div className="example-row">
                  {EXAMPLE_QUERIES.map((example) => (
                    <button
                      key={example}
                      type="button"
                      className="example-pill"
                      onClick={() => setQuery(example)}
                    >
                      {example}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>

        <aside className="panel panel-sidebar">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Detail Panel</p>
              <h2>{selectedNode?.label ?? "No node selected"}</h2>
            </div>
            {selectedNode && <span className={`type-chip type-${selectedNode.type}`}>{selectedNode.type}</span>}
          </div>

          <section className="detail-section">
            <h3>Node details</h3>
            {selectedNode ? (
              <>
                <p className="detail-copy">{describeNode(selectedNode)}</p>
                <dl className="meta-grid">
                  {Object.entries(selectedNode.metadata).map(([key, value]) => (
                    <div key={key} className="meta-row">
                      <dt>{formatLabel(key)}</dt>
                      <dd>{formatValue(value)}</dd>
                    </div>
                  ))}
                </dl>
              </>
            ) : (
              <p className="muted">Click a node to inspect its source metadata and connected neighbors.</p>
            )}
          </section>

          <section className="detail-section">
            <h3>Related nodes</h3>
            {connectedNodes.length > 0 ? (
              <div className="related-list">
                {connectedNodes.map((node) => (
                  <button
                    key={node.id}
                    type="button"
                    className="related-card"
                    onClick={() => setSelectedNodeId(node.id)}
                  >
                    <strong>{node.label}</strong>
                    <span>{node.type}</span>
                  </button>
                ))}
              </div>
            ) : (
              <p className="muted">Related nodes appear here once a graph neighborhood has been loaded.</p>
            )}
          </section>

          <section className="detail-section answer-section">
            <h3>Grounded answer</h3>
            {answer ? (
              <>
                <p className="answer-text">{answer.answer || "No grounded answer was generated."}</p>
                <div className="answer-meta">
                  <span>{answer.answer_mode}</span>
                  <span>{answer.backend}</span>
                  <span>{answer.timings.total_seconds.toFixed(2)}s total</span>
                </div>
                <div className="sources-list">
                  {answer.sources.map((source) => (
                    <article key={source.chunk_id} className="source-card">
                      <header>
                        <strong>{source.metadata.page as string || source.chunk_id}</strong>
                        <span>{source.score.toFixed(3)}</span>
                      </header>
                      <p>{source.text}</p>
                    </article>
                  ))}
                </div>
              </>
            ) : (
              <p className="muted">The answer panel fills after you submit a query.</p>
            )}
          </section>
        </aside>
      </main>

      <footer className="query-dock">
        <form className="query-form" onSubmit={handleSubmit}>
          <label className="query-label" htmlFor="query">
            Ask the graph
          </label>
          <textarea
            id="query"
            className="query-input"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            rows={2}
            placeholder="What do I know about vector databases?"
          />

          <div className="query-controls">
            <div className="inline-control">
              <span>Answer mode</span>
              <select value={answerMode} onChange={(event) => setAnswerMode(event.target.value as AnswerMode)}>
                {ANSWER_MODES.map((mode) => (
                  <option key={mode} value={mode}>
                    {mode}
                  </option>
                ))}
              </select>
            </div>

            <div className="query-actions">
              <span className="draft-hint">Drafting: {deferredQuery || "No prompt yet"}</span>
              <button type="submit" className="submit-button" disabled={isSubmitting}>
                {isSubmitting ? "Activating graph..." : "Run query"}
              </button>
            </div>
          </div>
        </form>

        <div className="dock-summary">
          <div>
            <p className="summary-label">Last query</p>
            <p className="summary-value">{lastSubmittedQuery || "No query submitted yet."}</p>
          </div>
          <div>
            <p className="summary-label">Top hits</p>
            {search && search.results.length > 0 ? (
              <div className="summary-list">
                {search.results.slice(0, 3).map((result) => (
                  <button
                    key={result.chunk_id}
                    type="button"
                    className="summary-chip"
                    onClick={() => setSelectedNodeId(`chunk::${result.chunk_id}`)}
                  >
                    {(result.metadata.section as string) || result.chunk_id}
                  </button>
                ))}
              </div>
            ) : (
              <p className="summary-value">No retrieval results yet.</p>
            )}
          </div>
        </div>
      </footer>

      {errorMessage && (
        <div className="error-toast" role="alert">
          {errorMessage}
        </div>
      )}
    </div>
  );
}

function buildElements(
  graph: GraphNeighborhoodResponse,
  selectedNodeId: string | null
): ElementDefinition[] {
  const nodeElements: ElementDefinition[] = graph.nodes.map((node) => ({
    data: {
      id: node.id,
      label: node.label,
      type: node.type,
      highlighted: node.highlighted ? 1 : 0,
      selected: selectedNodeId === node.id ? 1 : 0
    }
  }));

  const edgeElements: ElementDefinition[] = graph.edges.map((edge) => ({
    data: {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: edge.type,
      weight: edge.weight ?? 0
    }
  }));

  return [...nodeElements, ...edgeElements];
}

function relatedEdges(edges: GraphEdgePayload[], selectedNodeId: string | null): GraphEdgePayload[] {
  if (!selectedNodeId) {
    return [];
  }
  return edges.filter((edge) => edge.source === selectedNodeId || edge.target === selectedNodeId);
}

function relatedNodes(
  nodes: GraphNodePayload[],
  edges: GraphEdgePayload[],
  selectedNodeId: string | null
): GraphNodePayload[] {
  if (!selectedNodeId) {
    return [];
  }

  const relatedIds = new Set<string>();
  for (const edge of edges) {
    relatedIds.add(edge.source === selectedNodeId ? edge.target : edge.source);
  }

  return nodes.filter((node) => relatedIds.has(node.id));
}

function formatLabel(value: string): string {
  return value.replace(/_/g, " ");
}

function formatValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.join(" / ");
  }
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  return String(value);
}

function describeNode(node: GraphNodePayload): string {
  if (node.type === "document") {
    return "Document nodes anchor the retrieved neighborhood and show which source page the active chunks came from.";
  }
  return "Chunk nodes represent the retrieval-ready text units that ground the answer and connect through similarity edges.";
}

function MetricCard(props: { label: string; value: string; tone?: "ok" | "warn" }) {
  return (
    <div className={`metric-card ${props.tone === "warn" ? "metric-warn" : ""}`}>
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}
