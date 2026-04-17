export type GraphNodeType = "document" | "chunk";
export type GraphEdgeType = "belongs_to" | "similar_to";
export type AnswerMode = "concise" | "normal" | "detailed" | "bullet_summary" | "technical";

export interface SearchResultPayload {
  chunk_id: string;
  score: number;
  text: string;
  metadata: Record<string, unknown>;
}

export interface SearchResponse {
  query: string;
  result_count: number;
  results: SearchResultPayload[];
}

export interface AnswerTimingsPayload {
  embedding_seconds: number;
  retrieval_seconds: number;
  generation_seconds: number;
  total_seconds: number;
  first_token_seconds: number | null;
}

export interface AnswerResponse {
  question: string;
  answer: string;
  answer_mode: AnswerMode;
  generated: boolean;
  model: string | null;
  backend: "chroma" | "faiss";
  collection_name: string;
  prompt_path: string;
  sources: SearchResultPayload[];
  timings: AnswerTimingsPayload;
}

export interface GraphNodePayload {
  id: string;
  type: GraphNodeType;
  label: string;
  highlighted: boolean;
  metadata: Record<string, unknown>;
}

export interface GraphEdgePayload {
  id: string;
  source: string;
  target: string;
  type: GraphEdgeType;
  weight: number | null;
  metadata: Record<string, unknown>;
}

export interface GraphNeighborhoodResponse {
  query: string;
  result_count: number;
  seed_node_ids: string[];
  nodes: GraphNodePayload[];
  edges: GraphEdgePayload[];
}

export interface APIErrorPayload {
  detail?: string;
}
