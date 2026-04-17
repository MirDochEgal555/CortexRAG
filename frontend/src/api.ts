import type {
  APIErrorPayload,
  AnswerMode,
  AnswerResponse,
  GraphNeighborhoodResponse,
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

export function fetchGraphNeighborhood(query: string): Promise<GraphNeighborhoodResponse> {
  return postJSON<GraphNeighborhoodResponse>("/graph/neighborhood", {
    query,
    top_k: 5,
    candidate_k: 10
  });
}

export function fetchAnswer(query: string, answerMode: AnswerMode): Promise<AnswerResponse> {
  return postJSON<AnswerResponse>("/answer", {
    query,
    answer_mode: answerMode,
    top_k: 3,
    candidate_k: 10
  });
}

export function fetchSearch(query: string): Promise<SearchResponse> {
  return postJSON<SearchResponse>("/search", {
    query,
    top_k: 5,
    candidate_k: 10
  });
}
