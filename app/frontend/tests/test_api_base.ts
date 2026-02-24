type TestFn = () => Promise<void> | void;

function assert(condition: unknown, message: string): void {
  if (!condition) throw new Error(message);
}

function assertEqual<T>(actual: T, expected: T, label: string): void {
  if (actual !== expected) {
    throw new Error(`${label}: expected=${String(expected)} actual=${String(actual)}`);
  }
}

async function run(name: string, fn: TestFn): Promise<void> {
  try {
    await fn();
    console.log(`PASS ${name}`);
  } catch (err) {
    console.error(`FAIL ${name}`);
    throw err;
  }
}

class MemoryStorage implements Storage {
  private map = new Map<string, string>();

  get length(): number {
    return this.map.size;
  }

  clear(): void {
    this.map.clear();
  }

  getItem(key: string): string | null {
    return this.map.has(key) ? this.map.get(key)! : null;
  }

  key(index: number): string | null {
    return Array.from(this.map.keys())[index] ?? null;
  }

  removeItem(key: string): void {
    this.map.delete(key);
  }

  setItem(key: string, value: string): void {
    this.map.set(key, value);
  }
}

async function main(): Promise<void> {
  if (!("localStorage" in globalThis)) {
    (globalThis as { localStorage?: Storage }).localStorage = new MemoryStorage();
  }

  const base = await import("../src/api/base.js");
  const authStore = await import("../src/stores/authStore.js");

  const { request, ApiError } = base;
  const { useAuthStore } = authStore;

  const originalFetch = globalThis.fetch;

  await run("request adds bearer header", async () => {
    let authHeader = "";
    globalThis.fetch = (async (_input, init) => {
      authHeader = new Headers(init?.headers).get("Authorization") ?? "";
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }) as typeof fetch;

    const res = await request<{ ok: boolean }>("/purchases/my", { token: "token-123" });
    assertEqual(res.ok, true, "json response");
    assertEqual(authHeader, "Bearer token-123", "authorization header");
  });

  await run("request throws ApiError and clears auth on 401 with token", async () => {
    useAuthStore.setState({
      token: "persisted-token",
      user: {
        id: 1,
        username: "admin",
        name: "Admin",
        role: "admin",
        is_active: true,
      },
    });

    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ detail: "Not authenticated" }), {
        status: 401,
        headers: { "content-type": "application/json" },
      })) as typeof fetch;

    let caught: unknown;
    try {
      await request("/purchases/dashboard", { token: "persisted-token" });
    } catch (err) {
      caught = err;
    }

    assert(caught instanceof ApiError, "error type should be ApiError");
    assertEqual((caught as InstanceType<typeof ApiError>).status, 401, "error status");
    assertEqual((caught as Error).message, "Not authenticated", "error message");
    assertEqual(useAuthStore.getState().token, null, "token should be cleared");
    assertEqual(useAuthStore.getState().user, null, "user should be cleared");
  });

  await run("request keeps auth state on 401 without token option", async () => {
    useAuthStore.setState({
      token: "keep-me",
      user: {
        id: 2,
        username: "user",
        name: "User",
        role: "user",
        is_active: true,
      },
    });

    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ detail: "Not authenticated" }), {
        status: 401,
        headers: { "content-type": "application/json" },
      })) as typeof fetch;

    let caught: unknown;
    try {
      await request("/products");
    } catch (err) {
      caught = err;
    }

    assert(caught instanceof ApiError, "error type should be ApiError");
    assertEqual(useAuthStore.getState().token, "keep-me", "token should remain");
    assert(useAuthStore.getState().user !== null, "user should remain");
  });

  await run("request parses text error body fallback", async () => {
    globalThis.fetch = (async () =>
      new Response("Plain failure", {
        status: 500,
        headers: { "content-type": "text/plain" },
      })) as typeof fetch;

    let caught: unknown;
    try {
      await request("/products");
    } catch (err) {
      caught = err;
    }

    assert(caught instanceof ApiError, "error type should be ApiError");
    assertEqual((caught as Error).message, "Plain failure", "plain-text fallback message");
  });

  globalThis.fetch = originalFetch;
  console.log("test_api_base completed");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
