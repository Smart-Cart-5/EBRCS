export function isAdminRoutePath(pathname: string): boolean {
  return pathname === "/admin" || pathname.startsWith("/admin/");
}

export function shouldRedirectForAdminRoute(pathname: string, isAdminUser: boolean): boolean {
  return isAdminRoutePath(pathname) && !isAdminUser;
}
