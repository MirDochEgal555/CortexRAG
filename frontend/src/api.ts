import type {
  APIErrorPayload,
  AnswerMode,
  AnswerResponse,
  GraphNeighborhoodResponse,
  RetrievalFilters,
  SearchResponse
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

async function postJSON<TResponse>(path: string, payload: Record<string, unknown>): Promise<TResponse> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const errorPayload = (await response.json()) as APIErrorPayload;
      if (errorPayload.detail) {
        detail = errorPayload.detail;
      }
    } catch {
      // Keep the HTTP fallback message when the error body is not JSON.
    }
    throw new Error(detail);
  }

  return (await response.json()) as TResponse;
}

export async function getHealth(): Promise<{ status: string; service: string }> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed with ${response.status}.`);
  }
  return (await response.json()) as { status: string; service: string };
}

function withFilters(payload: Record<string, unknown>, filters: RetrievalFilters): Record<string, unknown> {
  return {
    ...payload,
    ...(filters.source ? { source: filters.source } : {}),
    ...(filters.document_id ? { document_id: filters.document_id } : {}),
    ...(filters.citekey ? { citekey: filters.citekey } : {}),
    ...(filters.doi ? { doi: filters.doi } : {}),
    ...(filters.title ? { title: filters.title } : {}),
    ...(filters.zotero_key ? { zotero_key: filters.zotero_key } : {}),
    ...(filters.min_score !== undefined ? { min_score: filters.min_score } : {})
  };
}

export function fetchGraphNeighborhood(
  query: string,
  filters: RetrievalFilters
): Promise<GraphNeighborhoodResponse> {
  return postJSON<GraphNeighborhoodResponse>("/graph/neighborhood", {
    ...withFilters(
      {
        query,
        top_k: 5,
        candidate_k: 10
      },
      filters
    )
  });
}

export function fetchAnswer(
  query: string,
  answerMode: AnswerMode,
  filters: RetrievalFilters
): Promise<AnswerResponse> {
  return postJSON<AnswerResponse>("/answer", {
    ...withFilters(
      {
        query,
        answer_mode: answerMode,
        top_k: 3,
        candidate_k: 10
      },
      filters
    )
  });
}

export function fetchSearch(query: string, filters: RetrievalFilters): Promise<SearchResponse> {
  return postJSON<SearchResponse>("/search", {
    ...withFilters(
      {
        query,
        top_k: 5,
        candidate_k: 10
      },
      filters
    )
  });
}
