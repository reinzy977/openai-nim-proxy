// server.js - OpenAI to NVIDIA NIM API Proxy (fixed version with logging)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Show the model's reasoning inside <think> tags (keep true so you see something while it thinks)
const SHOW_REASONING = true;
// Leave false (the guide says never turn this on)
const ENABLE_THINKING_MODE = false;

// Names you type in Janitor -> real NVIDIA model IDs (free ones only)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v4.1-flash',
  'gpt-4o': 'mistralai/mistral-large-4-1t-a52b',
  'gpt-4-turbo': 'z-ai/glm-5.3-flash',
  'gpt-4': 'minimaxai/minimax-m3.1-flash',
  'gemini-pro': 'google/gemma-4-31b-it',
  'gpt-oss': 'openai/gpt-oss-20b',
  'gpt-5': 'nvidia/nemotron-3-ultra-550b-a55b'
};

app.get('/health', (req, res) => {
  res.json({ status: 'ok', reasoning_display: SHOW_REASONING, thinking_mode: ENABLE_THINKING_MODE });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Read an error body even when it comes back as a stream
async function readErrorBody(error) {
  const data = error.response && error.response.data;
  if (!data) return '';
  if (typeof data.on === 'function') {
    return await new Promise(resolve => {
      let s = '';
      data.on('data', c => (s += c.toString()));
      data.on('end', () => resolve(s));
      data.on('error', () => resolve(s));
    });
  }
  return typeof data === 'string' ? data : JSON.stringify(data);
}

app.post('/v1/chat/completions', async (req, res) => {
  const started = Date.now();
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // Use the mapping if the name is in it, otherwise pass the name straight through
    const nimModel = MODEL_MAPPING[model] || model;
    console.log(`REQUEST: ${model} -> ${nimModel} | stream: ${!!stream}`);

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 4096,
      stream: !!stream
    };
    if (ENABLE_THINKING_MODE) {
      nimRequest.chat_template_kwargs = { thinking: true };
    }

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 60000
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.flushHeaders && res.flushHeaders();

      // Keep the connection alive while the model is thinking
      const keepAlive = setInterval(() => res.write(': keepalive\n\n'), 8000);

      let buffer = '';
      let reasoningStarted = false;
      let gotFirstChunk = false;

      response.data.on('data', (chunk) => {
        if (!gotFirstChunk) {
          gotFirstChunk = true;
          console.log(`FIRST CHUNK after ${Date.now() - started} ms`);
        }
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;
          if (line.includes('[DONE]')) {
            res.write('data: [DONE]\n\n');
            return;
          }
          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices && data.choices[0] && data.choices[0].delta;
            if (delta) {
              const reasoning = delta.reasoning_content || delta.reasoning;
              const content = delta.content;
              let out = '';

              if (SHOW_REASONING) {
                if (reasoning && !reasoningStarted) {
                  out = '<think>\n' + reasoning;
                  reasoningStarted = true;
                } else if (reasoning) {
                  out = reasoning;
                }
                if (content && reasoningStarted) {
                  out += '\n</think>\n\n' + content;
                  reasoningStarted = false;
                } else if (content) {
                  out += content;
                }
              } else if (content) {
                out = content;
              }

              delta.content = out;
              delete delta.reasoning_content;
              delete delta.reasoning;
            }
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            // ignore partial lines
          }
        });
      });

      response.data.on('end', () => {
        clearInterval(keepAlive);
        console.log(`STREAM DONE after ${Date.now() - started} ms`);
        res.end();
      });
      response.data.on('error', (err) => {
        clearInterval(keepAlive);
        console.error('STREAM ERROR:', err.message);
        res.end();
      });
      req.on('close', () => clearInterval(keepAlive));
    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = (choice.message && choice.message.content) || '';
          const reasoning = choice.message && (choice.message.reasoning_content || choice.message.reasoning);
          if (SHOW_REASONING && reasoning) {
            fullContent = '<think>\n' + reasoning + '\n</think>\n\n' + fullContent;
          }
          return {
            index: choice.index,
            message: { role: (choice.message && choice.message.role) || 'assistant', content: fullContent },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      console.log(`DONE (non-stream) after ${Date.now() - started} ms`);
      res.json(openaiResponse);
    }
  } catch (error) {
    const body = await readErrorBody(error);
    console.error(`PROXY ERROR after ${Date.now() - started} ms:`, error.message, '|', body);
    if (!res.headersSent) {
      res.status((error.response && error.response.status) || 500).json({
        error: {
          message: body || error.message || 'Internal server error',
          type: 'invalid_request_error',
          code: (error.response && error.response.status) || 500
        }
      });
    } else {
      res.end();
    }
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
});
