import { QueryClient } from "@tanstack/react-query";
import { ApiError } from "./api/base";

export function shouldRetryQueries(failureCount: number, error: unknown): boolean {
  if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
    return false;
  }
  return failureCount < 1;
}

export function createAppQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: shouldRetryQueries,
        staleTime: 30_000,
      },
    },
  });
}
