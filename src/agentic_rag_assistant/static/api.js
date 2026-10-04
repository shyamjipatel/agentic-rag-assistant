/** Same-origin JSON API. Credentials and provider selection stay on the server. */
export class ApiError extends Error {
  constructor(message, status = 0, requestId = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.requestId = requestId;
  }
}

export async function request(path, { method = "GET", body } = {}) {
  const options = { method, credentials: "same-origin", headers: {} };
  if (body instanceof FormData) options.body = body;
  else if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  let response;
  try { response = await fetch(path, options); }
  catch { throw new ApiError("Connection interrupted. Check that the application is running."); }
  if (response.status === 204) return null;
  let data;
  try { data = await response.json(); }
  catch { throw new ApiError("The server returned an unreadable response.", response.status); }
  if (!response.ok) {
    let detail = typeof data.detail === "string" ? data.detail : "The request could not be completed.";
    if (Array.isArray(data.detail)) {
      detail = data.detail.map(item => `${item.loc?.slice(1).join(".") || "Input"}: ${item.msg}`).join("; ");
    }
    const requestId = response.headers.get("X-Request-ID");
    throw new ApiError(`${detail}${requestId ? ` (Request ${requestId})` : ""}`, response.status, requestId);
  }
  return data;
}

export const api = {
  conversations: offset => request(`/conversations?limit=50&offset=${offset}`),
  create: () => request("/conversations", { method: "POST" }),
  history: (id, before) => request(`/conversations/${encodeURIComponent(id)}?limit=20${before ? `&before_turn=${before}` : ""}`),
  rename: (id, title) => request(`/conversations/${encodeURIComponent(id)}`, { method: "PATCH", body: { title } }),
  remove: id => request(`/conversations/${encodeURIComponent(id)}`, { method: "DELETE" }),
  documents: offset => request(`/documents?limit=50&offset=${offset}`),
  index: file => {
    const body = new FormData();
    body.append("file", file);
    return request("/documents/index", { method: "POST", body });
  },
  ask: body => request("/ask", { method: "POST", body }),
};
