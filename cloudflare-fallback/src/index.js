// Vorbe's last-resort AI fallback. Only reached from api_server.py if every
// OpenRouter model (primary + backup) AND Groq have failed - runs on
// Cloudflare's own infrastructure, independent of both.
export default {
  async fetch(request, env) {
    if (request.method !== "POST") {
      return new Response("Method not allowed", { status: 405 });
    }

    const auth = request.headers.get("Authorization");
    if (!env.WORKER_AUTH_TOKEN || auth !== `Bearer ${env.WORKER_AUTH_TOKEN}`) {
      return new Response("Unauthorized", { status: 401 });
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return new Response("Invalid JSON body", { status: 400 });
    }

    const { system, prompt, max_tokens } = body;
    if (!prompt) {
      return new Response("Missing 'prompt'", { status: 400 });
    }

    try {
      const result = await env.AI.run("@cf/qwen/qwen2.5-coder-32b-instruct", {
        messages: [
          {
            role: "system",
            content: system || "You are a helpful Luau programming assistant for the Vortex game platform.",
          },
          { role: "user", content: prompt },
        ],
        max_tokens: max_tokens || 800,
      });

      // OpenAI-shaped response so api_server.py parses it the same way as
      // the OpenRouter/Groq responses.
      return Response.json({
        choices: [{ message: { content: result.response } }],
      });
    } catch (err) {
      return new Response(`Workers AI error: ${err.message}`, { status: 502 });
    }
  },
};
