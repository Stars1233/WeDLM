# Copyright 2025 Tencent wechat. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Web Demo for WeDLM Inference
A simple Flask-based web interface for interacting with WeDLM models.
Supports command-line model path and runtime WeDLM parameter configuration.
Includes streaming output support for real-time token display.
"""
import os
import json
import argparse
from flask import Flask, request, jsonify, render_template_string, Response
from transformers import AutoTokenizer
from wedlm import LLM, SamplingParams
# ==================== Argument Parser ====================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WeDLM Web Demo")
    parser.add_argument(
        "--model-path",
        type=str,
        default="tencent/WeDLM-8B-Instruct",
        help="Path to the model directory",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    return parser.parse_args()
# ==================== Configuration ====================
# Default WeDLM decoding parameters
DEFAULT_WeDLM_ENTROPY_THRESHOLD = 0.4
DEFAULT_WeDLM_POS_PENALTY_FACTOR = 0.02
# Generation settings
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
# Engine settings
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.8
MAX_NUM_SEQS = 128
WeDLM_WINDOW_SIZE = 16
# ==================== Flask App ====================
app = Flask(__name__)
# Global variables for model components (initialized in main)
tokenizer = None
llm = None
stop_token_ids = []
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WeDLM Chat Interface</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <script>
    window.MathJax = {
      tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] },
      svg: { fontCache: 'global' }
    };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        :root {
            --bg-color: #f3f4f6;
            --chat-bg: #ffffff;
            --header-bg: #ffffff;
            --user-msg-bg: #2563eb;
            --user-text-color: #ffffff;
            --ai-msg-bg: #ffffff;
            --ai-text-color: #1f2937;
            --border-color: #e5e7eb;
            --stats-color: #64748b;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--ai-text-color);
            margin: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }
        header {
            background-color: var(--header-bg);
            padding: 15px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: var(--shadow-sm);
            z-index: 10;
        }
        header h1 {
            margin: 0;
            font-size: 1.15rem;
            font-weight: 700;
            color: #111827;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .header-right {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        header .status {
            font-size: 0.8rem;
            color: #10b981;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        /* Settings Bar - Always Visible */
        .settings-bar {
            background: var(--chat-bg);
            border-bottom: 1px solid var(--border-color);
            padding: 16px 24px;
            display: flex;
            flex-wrap: wrap;
            gap: 32px;
            align-items: center;
            box-shadow: var(--shadow-sm);
        }
        .setting-item {
            display: flex;
            flex-direction: column;
            gap: 6px;
            min-width: 280px;
            flex: 1;
            max-width: 400px;
        }
        .setting-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .setting-label {
            font-weight: 600;
            font-size: 0.85rem;
            color: #111827;
        }
        .setting-value {
            font-family: 'SF Mono', 'Menlo', monospace;
            font-size: 0.85rem;
            color: var(--user-msg-bg);
            font-weight: 600;
            background: #eff6ff;
            padding: 2px 8px;
            border-radius: 4px;
        }
        .setting-desc {
            font-size: 0.75rem;
            color: #6b7280;
            line-height: 1.4;
        }
        .setting-slider {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #e5e7eb;
            outline: none;
        }
        .setting-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--user-msg-bg);
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
            transition: transform 0.1s;
        }
        .setting-slider::-webkit-slider-thumb:hover {
            transform: scale(1.1);
        }
        .toggle-container {
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 180px;
        }
        .toggle-switch {
            position: relative;
            width: 48px;
            height: 26px;
            background: #d1d5db;
            border-radius: 13px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .toggle-switch.active {
            background: #10b981;
        }
        .toggle-switch::after {
            content: '';
            position: absolute;
            top: 3px;
            left: 3px;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            transition: transform 0.3s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .toggle-switch.active::after {
            transform: translateX(22px);
        }
        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            display: flex;
            flex-direction: column;
            gap: 24px;
            scroll-behavior: smooth;
            background-color: var(--bg-color);
        }
        .message {
            display: flex;
            flex-direction: column;
            max-width: 85%;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user { align-self: flex-end; align-items: flex-end; }
        .message.ai { align-self: flex-start; align-items: flex-start; }
        .bubble {
            padding: 14px 20px;
            border-radius: 16px;
            line-height: 1.6;
            position: relative;
            word-wrap: break-word;
            font-size: 0.95rem;
            box-shadow: var(--shadow-sm);
            tab-size: 4;
            -moz-tab-size: 4;
        }
        .message.user .bubble {
            background-color: var(--user-msg-bg);
            color: var(--user-text-color);
            border-bottom-right-radius: 4px;
        }
        .message.ai .bubble {
            background-color: var(--ai-msg-bg);
            color: var(--ai-text-color);
            border-bottom-left-radius: 4px;
            border: 1px solid var(--border-color);
        }
        .role-label {
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 6px;
            margin-left: 4px;
            margin-right: 4px;
            color: #6b7280;
        }
        .stats-box {
            margin-top: 8px;
            font-size: 0.75rem;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            background: #ffffff;
            padding: 6px 10px;
            border-radius: 6px;
            width: fit-content;
            border: 1px solid var(--border-color);
            color: var(--stats-color);
        }
        .stats-item { display: flex; align-items: center; gap: 5px; }
        .stats-label { font-weight: 500; }
        .stats-value {
            font-family: 'SF Mono', 'Menlo', monospace;
            font-weight: 600;
            color: #111827;
        }
        .token-debug-box {
            margin-top: 5px;
            font-size: 0.75rem;
            color: #6b7280;
            max-width: 100%;
        }
        .token-debug-box details summary {
            cursor: pointer;
            outline: none;
            user-select: none;
            padding: 4px 0;
        }
        .token-debug-box details summary:hover { color: #374151; }
        .token-list {
            font-family: 'SF Mono', 'Menlo', monospace;
            background: #ffffff;
            padding: 10px;
            border-radius: 6px;
            word-break: break-all;
            margin-top: 5px;
            border: 1px solid var(--border-color);
            white-space: pre-wrap;
            color: #374151;
            tab-size: 4;
            -moz-tab-size: 4;
        }
        .bubble pre {
            background: #f8fafc;
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid #e2e8f0;
            margin: 10px 0;
            white-space: pre;
            word-wrap: normal;
            tab-size: 4;
            -moz-tab-size: 4;
        }
        .bubble code {
            font-family: 'SF Mono', 'Menlo', Consolas, monospace;
            font-size: 0.9em;
            background-color: rgba(0,0,0,0.05);
            padding: 2px 4px;
            border-radius: 4px;
            white-space: pre-wrap;
            tab-size: 4;
            -moz-tab-size: 4;
        }
        .bubble pre code {
            background-color: transparent;
            padding: 0;
            color: inherit;
            white-space: pre;
            display: block;
        }
        .bubble p { margin: 0 0 10px 0; }
        .bubble p:last-child { margin: 0; }
        .message.user .bubble a { color: #fff; text-decoration: underline; }
        .message.ai .bubble a { color: var(--user-msg-bg); }
        #input-area {
            background-color: var(--chat-bg);
            padding: 20px 30px;
            border-top: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            gap: 15px;
            box-shadow: 0 -4px 6px -1px rgba(0,0,0,0.02);
        }
        #suggested-prompts-container { display: flex; flex-wrap: wrap; gap: 8px; }
        .prompt-suggestion {
            background-color: #f3f4f6;
            border: 1px solid transparent;
            color: #4b5563;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s ease;
            user-select: none;
            font-weight: 500;
        }
        .prompt-suggestion:hover { background-color: #e5e7eb; color: #111827; }
        #main-input-row {
            display: flex;
            gap: 12px;
            align-items: flex-end;
            width: 100%;
        }
        textarea {
            flex: 1;
            background-color: #ffffff;
            border: 1px solid #d1d5db;
            border-radius: 12px;
            color: #1f2937;
            padding: 14px;
            font-size: 1rem;
            resize: none;
            height: 24px;
            max-height: 200px;
            outline: none;
            transition: all 0.2s;
            font-family: inherit;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            tab-size: 4;
            -moz-tab-size: 4;
        }
        textarea:focus {
            border-color: var(--user-msg-bg);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        button {
            background-color: var(--user-msg-bg);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0 24px;
            height: 54px;
            cursor: pointer;
            font-weight: 600;
            transition: background-color 0.2s, transform 0.1s;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        button:hover { background-color: #1d4ed8; }
        button:active { transform: translateY(1px); }
        button:disabled { background-color: #9ca3af; cursor: not-allowed; transform: none; }
        .loading-dots::after {
            content: ' .';
            animation: dots 1.5s steps(5, end) infinite;
        }
        @keyframes dots {
            0%, 20% { color: rgba(0,0,0,0); text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0); }
            40% { color: #333; text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0); }
            60% { text-shadow: .25em 0 0 #333, .5em 0 0 rgba(0,0,0,0); }
            80%, 100% { text-shadow: .25em 0 0 #333, .5em 0 0 #333; }
        }
        .streaming-cursor {
            display: inline-block;
            width: 2px;
            height: 1em;
            background-color: var(--user-msg-bg);
            animation: blink 1s infinite;
            margin-left: 2px;
            vertical-align: text-bottom;
        }
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
    </style>
</head>
<body>
<header>
    <h1>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color: var(--user-msg-bg);">
            <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        WeDLM Chat
    </h1>
    <div class="header-right">
        <div class="status">● Online</div>
    </div>
</header>
<div class="settings-bar">
    <div class="setting-item">
        <div class="setting-header">
            <span class="setting-label">Temperature</span>
            <span class="setting-value" id="temperature-value">{{ default_temperature }}</span>
        </div>
        <input type="range" class="setting-slider" id="temperature-slider" 
               min="0" max="0.5" step="0.01" value="{{ default_temperature }}">
        <div class="setting-desc">
            Controls randomness in generation. Lower values make output more deterministic, 
            higher values increase creativity.
        </div>
    </div>
    <div class="setting-item">
        <div class="setting-header">
            <span class="setting-label">Entropy Threshold</span>
            <span class="setting-value" id="entropy-value">{{ default_entropy }}</span>
        </div>
        <input type="range" class="setting-slider" id="entropy-slider" 
               min="0.1" max="1.0" step="0.05" value="{{ default_entropy }}">
        <div class="setting-desc">
            Higher values increase generation speed but may reduce quality. 
            Controls when to accept parallel token predictions.
        </div>
    </div>
    <div class="setting-item">
        <div class="setting-header">
            <span class="setting-label">Position Penalty Factor</span>
            <span class="setting-value" id="penalty-value">{{ default_penalty }}</span>
        </div>
        <input type="range" class="setting-slider" id="penalty-slider"
               min="0.00" max="0.10" step="0.01" value="{{ default_penalty }}">
        <div class="setting-desc">
            Higher values improve quality but decrease speed. 
            Penalizes predictions at later positions in the window.
        </div>
    </div>
    <div class="toggle-container">
        <div class="toggle-switch active" id="stream-toggle" onclick="toggleStreaming()"></div>
        <span class="toggle-label">Streaming</span>
    </div>
</div>
<div id="chat-container">
    <div class="message ai">
        <div class="role-label">Assistant</div>
        <div class="bubble">Hello! I am the WeDLM model. How can I help you today?</div>
    </div>
</div>
<div id="input-area">
    <div id="suggested-prompts-container">
        <div class="prompt-suggestion" onclick="setPrompt('Count from 1 to 200')">Count from 1 to 200</div>
        <div class="prompt-suggestion" onclick="setPrompt('If 3x + 3 = 15, what is x?')">Math: Solve 3x+3=15</div>
        <div class="prompt-suggestion" onclick="setPrompt('Explain Quantum Physics simply.')">Explain Quantum Physics</div>
        <div class="prompt-suggestion" onclick="setPrompt('Write a Python program to solve sudoku.')">Python: Sudoku Solver</div>
    </div>
    <div id="main-input-row">
        <textarea id="user-input" placeholder="Type your message... (Shift+Enter for new line)" rows="1"></textarea>
        <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>
</div>
<script>
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const temperatureSlider = document.getElementById('temperature-slider');
    const entropySlider = document.getElementById('entropy-slider');
    const penaltySlider = document.getElementById('penalty-slider');
    const temperatureValue = document.getElementById('temperature-value');
    const entropyValue = document.getElementById('entropy-value');
    const penaltyValue = document.getElementById('penalty-value');
    const streamToggle = document.getElementById('stream-toggle');
    let streamingEnabled = true;
    marked.setOptions({ breaks: true, gfm: true, headerIds: false, mangle: false });
    temperatureSlider.addEventListener('input', () => {
        temperatureValue.textContent = parseFloat(temperatureSlider.value).toFixed(2);
    });
    entropySlider.addEventListener('input', () => {
        entropyValue.textContent = parseFloat(entropySlider.value).toFixed(2);
    });
    penaltySlider.addEventListener('input', () => {
        penaltyValue.textContent = parseFloat(penaltySlider.value).toFixed(2);
    });
    function toggleStreaming() {
        streamingEnabled = !streamingEnabled;
        streamToggle.classList.toggle('active', streamingEnabled);
    }
    function setPrompt(promptText) {
        userInput.value = promptText;
        userInput.dispatchEvent(new Event('input', { bubbles: true }));
        userInput.focus();
    }
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value === '') this.style.height = '24px';
    });
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    function processTabsInHtml(html) {
        // Preserve tabs in the HTML content
        return html.replace(/\t/g, '&#9;');
    }
    function addMessage(role, text, isHtml = false, tokenIds = null, stats = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        const label = document.createElement('div');
        label.className = 'role-label';
        label.innerText = role === 'user' ? 'You' : 'Assistant';
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        if (isHtml) {
            bubble.innerHTML = processTabsInHtml(text);
        } else {
            bubble.innerText = text;
        }
        msgDiv.appendChild(label);
        msgDiv.appendChild(bubble);
        if (stats && role === 'ai') {
            const statsBox = document.createElement('div');
            statsBox.className = 'stats-box';
            statsBox.innerHTML = `
                <div class="stats-item"><span class="stats-label">Prefill:</span><span class="stats-value">${(stats.prefill_speed || 0).toFixed(1)} t/s</span></div>
                <div class="stats-item"><span class="stats-label">Decode:</span><span class="stats-value">${(stats.decode_speed || 0).toFixed(1)} t/s</span></div>
                <div class="stats-item"><span class="stats-label">Tok/Fwd:</span><span class="stats-value" style="color:#d97706;">${(stats.tokens_per_forward || 1).toFixed(2)}</span></div>
                <div class="stats-item"><span class="stats-label">Tokens:</span><span class="stats-value" style="color:#059669;">${stats.decode_tokens || 0}</span></div>
                <div class="stats-item"><span class="stats-label">Forwards:</span><span class="stats-value" style="color:#7c3aed;">${stats.decode_forwards || 0}</span></div>
            `;
            msgDiv.appendChild(statsBox);
        }
        if (tokenIds && Array.isArray(tokenIds) && tokenIds.length > 0) {
            const debugBox = document.createElement('div');
            debugBox.className = 'token-debug-box';
            debugBox.innerHTML = `<details><summary>Token IDs (${tokenIds.length})</summary><div class="token-list">${JSON.stringify(tokenIds)}</div></details>`;
            msgDiv.appendChild(debugBox);
        }
        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        if (role === 'ai') {
            hljs.highlightAll();
            if (window.MathJax) MathJax.typesetPromise([bubble]);
        }
        return msgDiv;
    }
    function createStreamingMessage() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai';
        const label = document.createElement('div');
        label.className = 'role-label';
        label.innerText = 'Assistant';
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.id = 'streaming-bubble';
        
        const cursor = document.createElement('span');
        cursor.className = 'streaming-cursor';
        bubble.appendChild(cursor);
        msgDiv.appendChild(label);
        msgDiv.appendChild(bubble);
        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return msgDiv;
    }
    function updateStreamingMessage(msgDiv, text, stats = null, tokenIds = null) {
        const bubble = msgDiv.querySelector('.bubble');
        
        // Remove cursor if present
        const cursor = bubble.querySelector('.streaming-cursor');
        if (cursor) cursor.remove();
        // Update content
        bubble.innerHTML = processTabsInHtml(marked.parse(text));
        // Add cursor back for streaming effect
        const newCursor = document.createElement('span');
        newCursor.className = 'streaming-cursor';
        bubble.appendChild(newCursor);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    function finalizeStreamingMessage(msgDiv, text, stats = null, tokenIds = null) {
        const bubble = msgDiv.querySelector('.bubble');
        
        // Remove cursor
        const cursor = bubble.querySelector('.streaming-cursor');
        if (cursor) cursor.remove();
        // Final content update with markdown rendering
        bubble.innerHTML = processTabsInHtml(marked.parse(text));
        // Add stats if available
        if (stats) {
            const statsBox = document.createElement('div');
            statsBox.className = 'stats-box';
            statsBox.innerHTML = `
                <div class="stats-item"><span class="stats-label">Prefill:</span><span class="stats-value">${(stats.prefill_speed || 0).toFixed(1)} t/s</span></div>
                <div class="stats-item"><span class="stats-label">Decode:</span><span class="stats-value">${(stats.decode_speed || 0).toFixed(1)} t/s</span></div>
                <div class="stats-item"><span class="stats-label">Tok/Fwd:</span><span class="stats-value" style="color:#d97706;">${(stats.tokens_per_forward || 1).toFixed(2)}</span></div>
                <div class="stats-item"><span class="stats-label">Tokens:</span><span class="stats-value" style="color:#059669;">${stats.decode_tokens || 0}</span></div>
                <div class="stats-item"><span class="stats-label">Forwards:</span><span class="stats-value" style="color:#7c3aed;">${stats.decode_forwards || 0}</span></div>
            `;
            msgDiv.appendChild(statsBox);
        }
        // Add token IDs if available
        if (tokenIds && Array.isArray(tokenIds) && tokenIds.length > 0) {
            const debugBox = document.createElement('div');
            debugBox.className = 'token-debug-box';
            debugBox.innerHTML = `<details><summary>Token IDs (${tokenIds.length})</summary><div class="token-list">${JSON.stringify(tokenIds)}</div></details>`;
            msgDiv.appendChild(debugBox);
        }
        hljs.highlightAll();
        if (window.MathJax) MathJax.typesetPromise([bubble]);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    async function sendMessageStreaming(text) {
        addMessage('user', text);
        userInput.value = '';
        userInput.style.height = '24px';
        userInput.disabled = true;
        sendBtn.disabled = true;
        const msgDiv = createStreamingMessage();
        let fullText = '';
        let allTokenIds = [];
        let finalStats = null;
        try {
            const response = await fetch('/generate_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: text,
                    temperature: parseFloat(temperatureSlider.value),
                    entropy_threshold: parseFloat(entropySlider.value),
                    pos_penalty_factor: parseFloat(penaltySlider.value)
                })
            });
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                
                // Process complete SSE events
                const lines = buffer.split('\\n');
                buffer = lines.pop() || '';  // Keep incomplete line in buffer
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const jsonStr = line.slice(6);
                        if (jsonStr.trim() === '') continue;
                        
                        try {
                            const data = JSON.parse(jsonStr);
                            
                            if (data.new_text) {
                                fullText += data.new_text;
                                updateStreamingMessage(msgDiv, fullText);
                            }
                            
                            if (data.new_token_ids && data.new_token_ids.length > 0) {
                                allTokenIds = allTokenIds.concat(data.new_token_ids);
                            }
                            
                            if (data.is_finished && data.stats) {
                                finalStats = data.stats;
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e, jsonStr);
                        }
                    }
                }
            }
            // Finalize the message
            finalizeStreamingMessage(msgDiv, fullText, finalStats, allTokenIds);
        } catch (err) {
            const bubble = msgDiv.querySelector('.bubble');
            bubble.innerHTML = "Network Error: " + err.message;
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    }
    async function sendMessageNonStreaming(text) {
        addMessage('user', text);
        userInput.value = '';
        userInput.style.height = '24px';
        userInput.disabled = true;
        sendBtn.disabled = true;
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message ai';
        loadingDiv.innerHTML = `<div class="role-label">Assistant</div><div class="bubble"><span class="loading-dots">Thinking</span></div>`;
        chatContainer.appendChild(loadingDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: text,
                    temperature: parseFloat(temperatureSlider.value),
                    entropy_threshold: parseFloat(entropySlider.value),
                    pos_penalty_factor: parseFloat(penaltySlider.value)
                })
            });
            const data = await response.json();
            chatContainer.removeChild(loadingDiv);
            if (data.error) {
                addMessage('ai', "Error: " + data.error);
            } else {
                addMessage('ai', marked.parse(data.response), true, data.token_ids, data.stats);
            }
        } catch (err) {
            chatContainer.removeChild(loadingDiv);
            addMessage('ai', "Network Error: " + err.message);
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    }
    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;
        if (streamingEnabled) {
            await sendMessageStreaming(text);
        } else {
            await sendMessageNonStreaming(text);
        }
    }
</script>
</body>
</html>
"""
@app.route("/")
def index():
    """Serve the main chat interface with default parameter values."""
    return render_template_string(
        HTML_TEMPLATE,
        default_temperature=DEFAULT_TEMPERATURE,
        default_entropy=DEFAULT_WeDLM_ENTROPY_THRESHOLD,
        default_penalty=DEFAULT_WeDLM_POS_PENALTY_FACTOR,
    )
@app.route("/generate", methods=["POST"])
def generate():
    """Generate response for the given prompt with configurable WeDLM parameters."""
    data = request.json
    user_prompt = data.get("prompt", "")
    if not user_prompt:
        return jsonify({"error": "Empty prompt"}), 400
    # Get temperature from request, with default
    temperature = data.get("temperature", DEFAULT_TEMPERATURE)
    # Get WeDLM parameters from request, with defaults
    entropy_threshold = data.get("entropy_threshold", DEFAULT_WeDLM_ENTROPY_THRESHOLD)
    pos_penalty_factor = data.get("pos_penalty_factor", DEFAULT_WeDLM_POS_PENALTY_FACTOR)
    try:
        # Format prompt with chat template
        messages = [{"role": "user", "content": user_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"Input Prompt: {formatted_prompt!r}")
        print(f"Temperature: {temperature}")
        print(f"WeDLM Params: entropy={entropy_threshold}, penalty={pos_penalty_factor}")
        # Create sampling params with user-specified settings
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=DEFAULT_TOP_P,
            top_k=DEFAULT_TOP_K,
            max_tokens=DEFAULT_MAX_TOKENS,
            stop_token_ids=stop_token_ids,
            wedlm_entropy_threshold=entropy_threshold,
            wedlm_pos_penalty_factor=pos_penalty_factor,
        )
        # Generate response
        outputs = llm.generate([formatted_prompt], sampling_params, use_tqdm=False)
        output_item = outputs[0]
        generated_text = output_item["text"]
        print(generated_text)
        token_ids = output_item.get("token_ids", [])
        stats = output_item.get("stats", {})
        return jsonify({
            "response": generated_text,
            "token_ids": token_ids,
            "stats": stats,
        })
    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({"error": str(e)}), 500
@app.route("/generate_stream", methods=["POST"])
def generate_stream():
    """Generate response with streaming output using Server-Sent Events."""
    data = request.json
    user_prompt = data.get("prompt", "")
    if not user_prompt:
        return jsonify({"error": "Empty prompt"}), 400
    # Get temperature from request, with default
    temperature = data.get("temperature", DEFAULT_TEMPERATURE)
    # Get WeDLM parameters from request, with defaults
    entropy_threshold = data.get("entropy_threshold", DEFAULT_WeDLM_ENTROPY_THRESHOLD)
    pos_penalty_factor = data.get("pos_penalty_factor", DEFAULT_WeDLM_POS_PENALTY_FACTOR)
    def generate_events():
        try:
            # Format prompt with chat template
            messages = [{"role": "user", "content": user_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            print(f"[Stream] Input Prompt: {formatted_prompt!r}")
            print(f"[Stream] Temperature: {temperature}")
            print(f"[Stream] WeDLM Params: entropy={entropy_threshold}, penalty={pos_penalty_factor}")
            # Create sampling params with user-specified settings
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=DEFAULT_TOP_P,
                top_k=DEFAULT_TOP_K,
                max_tokens=DEFAULT_MAX_TOKENS,
                stop_token_ids=stop_token_ids,
                wedlm_entropy_threshold=entropy_threshold,
                wedlm_pos_penalty_factor=pos_penalty_factor,
            )
            # Stream tokens using generate_stream
            for update in llm.generate_stream([formatted_prompt], sampling_params):
                event_data = json.dumps(update)
                yield f"data: {event_data}\n\n"
        except Exception as e:
            print(f"[Stream] Error during generation: {e}")
            error_data = json.dumps({"error": str(e), "is_finished": True})
            yield f"data: {error_data}\n\n"
    return Response(
        generate_events(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
def init_model(model_path: str):
    """Initialize the tokenizer and LLM engine."""
    global tokenizer, llm, stop_token_ids
    model_path = os.path.expanduser(model_path)
    print(f"Loading Tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Loading LLM Engine from {model_path}...")
    llm = LLM(
        model_path,
        enforce_eager=False,
        tensor_parallel_size=1,
        wedlm_window_size=WeDLM_WINDOW_SIZE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=MAX_NUM_SEQS,
    )
    # Build stop token IDs list
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    extra_stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    for token_str in extra_stop_tokens:
        if token_str in tokenizer.get_vocab():
            tid = tokenizer.convert_tokens_to_ids(token_str)
            if tid not in stop_token_ids:
                stop_token_ids.append(tid)
if __name__ == "__main__":
    args = parse_args()
    # Initialize model with command-line path
    init_model(args.model_path)
    print(f"Starting server on http://{args.host}:{args.port}")
    print(f"Default Temperature: {DEFAULT_TEMPERATURE}")
    print(f"Default WeDLM Entropy Threshold: {DEFAULT_WeDLM_ENTROPY_THRESHOLD}")
    print(f"Default WeDLM Position Penalty Factor: {DEFAULT_WeDLM_POS_PENALTY_FACTOR}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)