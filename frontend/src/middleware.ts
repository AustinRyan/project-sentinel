import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // In production (Vercel), redirect /dashboard to landing page.
  // Dashboard is only accessible via the local dev server.
  if (
    process.env.NODE_ENV === "production" &&
    pathname.startsWith("/dashboard")
  ) {
    return NextResponse.redirect(new URL("/", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/dashboard/:path*"],
};
