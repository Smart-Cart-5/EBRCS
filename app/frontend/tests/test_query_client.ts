import { ApiError } from "../src/api/base.js";
import { createAppQueryClient, shouldRetryQueries } from "../src/queryClient.js";

function assert(condition: unknown, message: string): void {
  if (!condition) throw new Error(message);
}

function assertEqual<T>(actual: T, expected: T, label: string): void {
  if (actual !== expected) {
    throw new Error(`${label}: expected=${String(expected)} actual=${String(actual)}`);
  }
}

function main(): void {
  assertEqual(shouldRetryQueries(0, new ApiError(401, "no auth")), false, "401 retry");
  assertEqual(shouldRetryQueries(0, new ApiError(403, "forbidden")), false, "403 retry");
  assertEqual(shouldRetryQueries(0, new ApiError(500, "server error")), true, "500 first retry");
  assertEqual(shouldRetryQueries(1, new ApiError(500, "server error")), false, "500 second retry");
  assertEqual(shouldRetryQueries(0, new Error("other")), true, "generic error first retry");
  assertEqual(shouldRetryQueries(1, new Error("other")), false, "generic error second retry");

  const queryClient = createAppQueryClient();
  const retryOption = queryClient.getDefaultOptions().queries?.retry;
  assert(typeof retryOption === "function", "query retry option should be function");

  const retryFn = retryOption as (failureCount: number, error: unknown) => boolean;
  assertEqual(retryFn(0, new ApiError(401, "no auth")), false, "retry fn 401");
  assertEqual(retryFn(0, new Error("network")), true, "retry fn network first");
  assertEqual(retryFn(1, new Error("network")), false, "retry fn network second");

  console.log("test_query_client completed");
}

main();
