// /v/<label>/<path> serves the frozen snapshot deployed at <label>.pickems.pages.dev (scripts/deploy.sh
// --snapshot), so every day's forecast lives under the site's own domain. Only labels listed in this
// deployment's versions.json are served, so the path can't reach other preview branches.
const LABEL = /^[a-z0-9-]{1,28}$/, PROJECT = 'pickems';

export async function onRequest({ request, params, env }) {
	const [label, ...rest] = params.path || [];
	const url = new URL(request.url);
	if (!label || !LABEL.test(label)) return new Response('Not found', { status: 404 });
	// Pages relative links resolve against the directory: /v/<label> needs its trailing slash.
	if (!rest.length && !url.pathname.endsWith('/')) return Response.redirect(`${url.origin}/v/${label}/${url.search}`, 308);

	const versions = await env.ASSETS.fetch(new URL('/versions.json', url)).then(r => r.ok ? r.json() : null).catch(() => null);
	if (!versions || !(versions.snapshots || []).some(s => s.label === label)) return new Response('No such snapshot', { status: 404 });

	const upstream = new URL(`https://${label}.${PROJECT}.pages.dev/${rest.map(encodeURIComponent).join('/')}${url.pathname.endsWith('/') && rest.length ? '/' : ''}${url.search}`);
	// Forward only cache/range headers: the snapshot host is public and needs nothing else.
	const headers = new Headers();
	for (const h of ['accept', 'range', 'if-none-match', 'if-modified-since']) if (request.headers.has(h)) headers.set(h, request.headers.get(h));
	const res = await fetch(upstream, { method: request.method, headers, redirect: 'manual' });

	// Upstream redirects (e.g. runs/<name>.html -> runs/<name>) point at its root: keep them under /v/<label>.
	const loc = res.headers.get('location');
	if (loc) {
		const target = new URL(loc, upstream);
		if (target.host === upstream.host) {
			const headers = new Headers(res.headers);
			headers.set('location', `/v/${label}${target.pathname}${target.search}`);
			return new Response(null, { status: res.status, headers });
		}
	}
	return res;
}
