import { isAdminRoutePath, shouldRedirectForAdminRoute } from "../src/routing/guards.js";

function assertEqual<T>(actual: T, expected: T, label: string): void {
  if (actual !== expected) {
    throw new Error(`${label}: expected=${String(expected)} actual=${String(actual)}`);
  }
}

function main(): void {
  assertEqual(isAdminRoutePath("/admin"), true, "admin root");
  assertEqual(isAdminRoutePath("/admin/purchases"), true, "admin child");
  assertEqual(isAdminRoutePath("/administrator"), false, "not admin route prefix");
  assertEqual(isAdminRoutePath("/"), false, "root route");

  assertEqual(shouldRedirectForAdminRoute("/admin", false), true, "redirect non-admin /admin");
  assertEqual(
    shouldRedirectForAdminRoute("/admin/purchases", false),
    true,
    "redirect non-admin child"
  );
  assertEqual(
    shouldRedirectForAdminRoute("/admin/purchases", true),
    false,
    "allow admin child"
  );
  assertEqual(shouldRedirectForAdminRoute("/products", false), false, "allow user route");

  console.log("test_route_guards completed");
}

main();
