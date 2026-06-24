(function () {
  // ============================================================================
  // Self-Attention Interactive Demo
  // ----------------------------------------------------------------------------
  // Mount point expected in the Quarto post:
  //
  // <div id="attn-demo"></div>
  // <script src="attention-demo.js"></script>
  //
  // This file builds the UI, extracts image patches, computes:
  //
  // Q = X W_Q^T
  // K = X W_K^T
  // V = X W_V^T
  // S = Q K^T / sqrt(d_k)
  // A = softmax(S) row-wise
  // Y = A V
  //
  // The weights are random. This is a mechanics demo, not a learned attention demo.
  // ============================================================================

  // --------------------------------------------------------------------------
  // Mount point
  // --------------------------------------------------------------------------

  const root = document.getElementById("attn-demo");

  if (!root) {
    console.warn("Could not find #attn-demo.");
    return;
  }

  // --------------------------------------------------------------------------
  // UI
  // --------------------------------------------------------------------------

  root.innerHTML = `
    <div style="
      font-family: Georgia, serif;
      background: #0f1117;
      border-radius: 12px;
      padding: 24px;
      color: #e0e0e0;
      max-width: 100%;
      box-sizing: border-box;
    ">

      <div style="margin-bottom: 20px;">
        <div style="
          font-size: 13px;
          color: #888;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          margin-bottom: 6px;
        ">
          Interactive Demo
        </div>

        <div style="
          font-size: 18px;
          font-weight: 600;
          color: #f0f0f0;
        ">
          Multi-Head Self-Attention Forward Pass
        </div>

        <div style="
          font-size: 13px;
          color: #999;
          margin-top: 4px;
        ">
          Click any patch to see which other patches it attends to.
          Weights are random - <em>not learned</em>.
        </div>

        <div id="attn-image-meta" style="
          margin-top: 10px;
          font-size: 12px;
          color: #777;
          line-height: 1.6;
        ">
          No image loaded yet.
        </div>
        <div id="attn-row-label" style="display:none;"></div>
      </div>

      <div style="
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 18px;
        align-items: center;
      ">
        <label style="font-size: 13px; color: #aaa;">
          Upload image:
          <input
            type="file"
            id="attn-upload"
            accept="image/*"
            style="
              margin-left: 8px;
              font-size: 12px;
              color: #ccc;
            "
          >
        </label>

        <label style="font-size: 13px; color: #aaa;">
          Patch size:
          <select
            id="attn-patch-size"
            style="
              margin-left: 6px;
              background: #1e2130;
              color: #ddd;
              border: 1px solid #333;
              border-radius: 4px;
              padding: 3px 6px;
              font-size: 12px;
            "
          >
            <option value="8">8x8</option>
            <option value="16">16x16</option>
            <option value="32" selected>32x32</option>
            <option value="56">56x56</option>
          </select>
        </label>

        <label style="font-size: 13px; color: #aaa; display:none;">
          Heads:
          <select
            id="attn-num-heads"
            style="
              margin-left: 6px;
              background: #1e2130;
              color: #ddd;
              border: 1px solid #333;
              border-radius: 4px;
              padding: 3px 6px;
              font-size: 12px;
            "
          >
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="4" selected>4</option>
            <option value="8">8</option>
          </select>
        </label>

        <label style="font-size: 13px; color: #aaa;">
          d<sub>k</sub>:
          <select
            id="attn-dk"
            style="
              margin-left: 6px;
              background: #1e2130;
              color: #ddd;
              border: 1px solid #333;
              border-radius: 4px;
              padding: 3px 6px;
              font-size: 12px;
            "
          >
            <option value="8">8</option>
            <option value="16">16</option>
            <option value="32" selected>32</option>
            <option value="64">64</option>
          </select>
        </label>

        <button
          id="attn-rerandom"
          style="
            background: #2a2f45;
            color: #ddd;
            border: 1px solid #444;
            border-radius: 6px;
            padding: 5px 14px;
            font-size: 12px;
            cursor: pointer;
          "
        >
          New random weights
        </button>
      </div>

      <div style="
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
        justify-content: center;
        align-items: flex-start;
      ">
        <div style="text-align: center; flex: 0 0 auto;">
          <div style="font-size: 12px; color: #888; margin-bottom: 6px;">
            Input image + patches
          </div>
          <canvas
            id="attn-canvas-img"
            style="
              border-radius: 6px;
              cursor: crosshair;
              border: 1px solid #333;
            "
          ></canvas>
        </div>

        <div style="flex: 1 1 560px; min-width: 320px;">
          <div style="font-size: 12px; color: #888; margin-bottom: 8px; text-align: center;">
            Attention views for all heads
          </div>
          <div id="attn-head-gallery" style="
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
          ">
            <div id="attn-head-card-0" style="background:#171a28;border:1px solid #2a2f45;border-radius:10px;padding:12px;display:none;">
              <div style="font-size:12px;color:#ddd;margin-bottom:8px;text-align:center;">Head 1</div>
              <div style="display:grid;gap:10px;justify-items:center;">
                <div>
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Attention heatmap on image</div>
                  <canvas id="attn-heat-0" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
                <div style="display:none;">
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Selected row of A</div>
                  <canvas id="attn-row-0" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
              </div>
            </div>
            <div id="attn-head-card-1" style="background:#171a28;border:1px solid #2a2f45;border-radius:10px;padding:12px;display:none;">
              <div style="font-size:12px;color:#ddd;margin-bottom:8px;text-align:center;">Head 2</div>
              <div style="display:grid;gap:10px;justify-items:center;">
                <div>
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Attention heatmap on image</div>
                  <canvas id="attn-heat-1" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
                <div style="display:none;">
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Selected row of A</div>
                  <canvas id="attn-row-1" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
              </div>
            </div>
            <div id="attn-head-card-2" style="background:#171a28;border:1px solid #2a2f45;border-radius:10px;padding:12px;display:none;">
              <div style="font-size:12px;color:#ddd;margin-bottom:8px;text-align:center;">Head 3</div>
              <div style="display:grid;gap:10px;justify-items:center;">
                <div>
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Attention heatmap on image</div>
                  <canvas id="attn-heat-2" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
                <div style="display:none;">
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Selected row of A</div>
                  <canvas id="attn-row-2" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
              </div>
            </div>
            <div id="attn-head-card-3" style="background:#171a28;border:1px solid #2a2f45;border-radius:10px;padding:12px;display:none;">
              <div style="font-size:12px;color:#ddd;margin-bottom:8px;text-align:center;">Head 4</div>
              <div style="display:grid;gap:10px;justify-items:center;">
                <div>
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Attention heatmap on image</div>
                  <canvas id="attn-heat-3" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
                <div style="display:none;">
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Selected row of A</div>
                  <canvas id="attn-row-3" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
              </div>
            </div>
            <div id="attn-head-card-4" style="background:#171a28;border:1px solid #2a2f45;border-radius:10px;padding:12px;display:none;">
              <div style="font-size:12px;color:#ddd;margin-bottom:8px;text-align:center;">Head 5</div>
              <div style="display:grid;gap:10px;justify-items:center;">
                <div>
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Attention heatmap on image</div>
                  <canvas id="attn-heat-4" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
                <div style="display:none;">
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Selected row of A</div>
                  <canvas id="attn-row-4" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
              </div>
            </div>
            <div id="attn-head-card-5" style="background:#171a28;border:1px solid #2a2f45;border-radius:10px;padding:12px;display:none;">
              <div style="font-size:12px;color:#ddd;margin-bottom:8px;text-align:center;">Head 6</div>
              <div style="display:grid;gap:10px;justify-items:center;">
                <div>
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Attention heatmap on image</div>
                  <canvas id="attn-heat-5" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
                <div style="display:none;">
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Selected row of A</div>
                  <canvas id="attn-row-5" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
              </div>
            </div>
            <div id="attn-head-card-6" style="background:#171a28;border:1px solid #2a2f45;border-radius:10px;padding:12px;display:none;">
              <div style="font-size:12px;color:#ddd;margin-bottom:8px;text-align:center;">Head 7</div>
              <div style="display:grid;gap:10px;justify-items:center;">
                <div>
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Attention heatmap on image</div>
                  <canvas id="attn-heat-6" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
                <div style="display:none;">
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Selected row of A</div>
                  <canvas id="attn-row-6" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
              </div>
            </div>
            <div id="attn-head-card-7" style="background:#171a28;border:1px solid #2a2f45;border-radius:10px;padding:12px;display:none;">
              <div style="font-size:12px;color:#ddd;margin-bottom:8px;text-align:center;">Head 8</div>
              <div style="display:grid;gap:10px;justify-items:center;">
                <div>
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Attention heatmap on image</div>
                  <canvas id="attn-heat-7" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
                <div style="display:none;">
                  <div style="font-size:11px;color:#888;margin-bottom:4px;text-align:center;">Selected row of A</div>
                  <canvas id="attn-row-7" style="border-radius:6px;border:1px solid #333;display:block;"></canvas>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div
        id="attn-info"
        style="
          margin-top: 18px;
          background: #1a1d2e;
          border-radius: 8px;
          padding: 14px 18px;
          font-size: 13px;
          color: #bbb;
          line-height: 1.7;
        "
      >
        <em>Load an image and click a patch to begin.</em>
      </div>

      <div style="
        margin-top: 16px;
        background: #12151f;
        border-left: 3px solid #4a7fb5;
        border-radius: 0 6px 6px 0;
        padding: 12px 16px;
        font-size: 12px;
        color: #888;
        line-height: 1.8;
      ">
        <strong style="color: #aaa;">What is being computed:</strong><br>
        For each head h: Q^(h), K^(h), V^(h), A^(h), Y^(h). Then concatenate heads column-wise and project back.<br>
        Q = X*W<sub>Q</sub><sup>T</sup>
        &nbsp;|&nbsp;
        K = X*W<sub>K</sub><sup>T</sup>
        &nbsp;|&nbsp;
        V = X*W<sub>V</sub><sup>T</sup><br>
        S = Q*K<sup>T</sup> / sqrt(d<sub>k</sub>)
        &nbsp;|&nbsp;
        A = softmax(S) row-wise
        &nbsp;|&nbsp;
        Y = A*V
      </div>
    </div>
  `;

  // --------------------------------------------------------------------------
  // DOM references
  const imageMeta = root.querySelector("#attn-image-meta");

  const heatmapLabel = root.querySelector("#attn-heatmap-label");
  const rowLabel = root.querySelector("#attn-row-label");

  const canvasImg = root.querySelector("#attn-canvas-img");
  const canvasHeat = root.querySelector("#attn-canvas-heat") || document.createElement("canvas");
  const canvasAttnRow = root.querySelector("#attn-canvas-attn-row") || document.createElement("canvas");

  const ctxImg = canvasImg.getContext("2d");
  const ctxHeat = canvasHeat.getContext("2d");
  const ctxAttnRow = canvasAttnRow.getContext("2d");

  const infoPanel = root.querySelector("#attn-info");

  const uploadInput = root.querySelector("#attn-upload");
  const patchSizeSelect = root.querySelector("#attn-patch-size");
  const numHeadsSelect = root.querySelector("#attn-num-heads");
  const headSelect = root.querySelector("#attn-head");
  const dkSelect = root.querySelector("#attn-dk");
  const rerandomButton = root.querySelector("#attn-rerandom");
  const headCards = Array.from({ length: 8 }, (_, h) => root.querySelector("#attn-head-card-" + h));
  const headHeatCanvases = Array.from({ length: 8 }, (_, h) => root.querySelector("#attn-heat-" + h));
  const headRowCanvases = Array.from({ length: 8 }, (_, h) => root.querySelector("#attn-row-" + h));
  const headHeatCtxs = headHeatCanvases.map((c) => c.getContext("2d"));
  const headRowCtxs = headRowCanvases.map((c) => c.getContext("2d"));

  // --------------------------------------------------------------------------
  // State

  const state = {
    img: null,

    originalW: 0,
    originalH: 0,

    processedW: 0,
    processedH: 0,

    usedW: 0,
    usedH: 0,

    X: null,

    N: 0,
    D: 0,
    H: 4,
    dk: 32,
    dv: 32,

    patchSize: 32,

    gridW: 0,
    gridH: 0,

    WQ_T_heads: [],
    WK_T_heads: [],
    WV_T_heads: [],
    W_O_T: null,

    Q_heads: [],
    K_heads: [],
    V_heads: [],
    S_heads: [],
    A_heads: [],
    Y_heads: [],
    Y_concat: null,
    Y_final: null,

    selectedPatch: null,
    selectedHead: 0,

    canvasW: 224,
    canvasH: 224,

    maxProcessingSide: 256,
    maxDisplaySide: 280
  };

  function updateHeadSelectOptions() {
    numHeadsSelect.value = String(state.H);
    headSelect.innerHTML = "";

    for (let h = 0; h < state.H; h++) {
      const option = document.createElement("option");
      option.value = String(h);
      option.textContent = `Head ${h + 1}`;
      headSelect.appendChild(option);
    }

    const nextHead = Math.min(Math.max(0, state.selectedHead), state.H - 1);
    state.selectedHead = nextHead;
    headSelect.value = String(nextHead);
    updateHeadLabels();
  }

  function updateHeadLabels() {
    const headNumber = state.selectedHead + 1;
    heatmapLabel.textContent = `Attention heatmap on image - Head ${headNumber}`;
    rowLabel.textContent = `Selected row of attention matrix A - Head ${headNumber}`;
  }

  function loadImageFromFile(file) {
    const img = new Image();
    const objectUrl = URL.createObjectURL(file);

    img.onload = function () {
      URL.revokeObjectURL(objectUrl);
      setupImage(img);
    };

    img.onerror = function () {
      URL.revokeObjectURL(objectUrl);
      infoPanel.innerHTML = `
        <span style="color:#f66;">Could not load the selected image.</span>
      `;
    };

    img.src = objectUrl;
  }

  function loadPersistedUploadIfPresent() {
    const file = uploadInput.files && uploadInput.files[0];
    if (!file) return false;

    loadImageFromFile(file);
    return true;
  }


  // --------------------------------------------------------------------------
  // Math helpers
  function syncHeadCardVisibility() {
    for (let i = 0; i < headCards.length; i++) {
      if (!headCards[i]) continue;
      headCards[i].style.display = i < state.H ? "block" : "none";
    }
  }
  // --------------------------------------------------------------------------

  function matmul(A, B, m, k, n) {
    const C = new Float32Array(m * n);

    for (let i = 0; i < m; i++) {
      const rowOffsetA = i * k;
      const rowOffsetC = i * n;

      for (let j = 0; j < n; j++) {
        let s = 0;

        for (let p = 0; p < k; p++) {
          s += A[rowOffsetA + p] * B[p * n + j];
        }

        C[rowOffsetC + j] = s;
      }
    }

    return C;
  }

  function transpose(A, m, n) {
    const T = new Float32Array(n * m);

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        T[j * m + i] = A[i * n + j];
      }
    }

    return T;
  }

  function softmaxRows(S, N) {
    const A = new Float32Array(N * N);

    for (let i = 0; i < N; i++) {
      const rowOffset = i * N;

      let maxVal = -Infinity;

      for (let j = 0; j < N; j++) {
        const v = S[rowOffset + j];
        if (v > maxVal) maxVal = v;
      }

      let sum = 0;

      for (let j = 0; j < N; j++) {
        const e = Math.exp(S[rowOffset + j] - maxVal);
        A[rowOffset + j] = e;
        sum += e;
      }

      for (let j = 0; j < N; j++) {
        A[rowOffset + j] /= sum;
      }
    }

    return A;
  }

  function randn() {
    let u = 0;
    let v = 0;

    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();

    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function xavier(rows, cols) {
    const W = new Float32Array(rows * cols);
    const scale = Math.sqrt(2 / (rows + cols));

    for (let i = 0; i < W.length; i++) {
      W[i] = randn() * scale;
    }

    return W;
  }

  // --------------------------------------------------------------------------
  // Image preprocessing
  // --------------------------------------------------------------------------

  function getProcessingSize(originalW, originalH, maxSide) {
    const maxOriginalSide = Math.max(originalW, originalH);

    if (maxOriginalSide <= maxSide) {
      return {
        processedW: originalW,
        processedH: originalH
      };
    }

    const scale = maxSide / maxOriginalSide;

    return {
      processedW: Math.max(1, Math.round(originalW * scale)),
      processedH: Math.max(1, Math.round(originalH * scale))
    };
  }

  function extractPatches(imgEl, patchSize) {
    const originalW = imgEl.naturalWidth;
    const originalH = imgEl.naturalHeight;

    const processingSize = getProcessingSize(
      originalW,
      originalH,
      state.maxProcessingSide
    );

    const processedW = processingSize.processedW;
    const processedH = processingSize.processedH;

    const gridW = Math.floor(processedW / patchSize);
    const gridH = Math.floor(processedH / patchSize);

    if (gridW < 1 || gridH < 1) {
      throw new Error("Patch size is larger than the processed image.");
    }

    const usedW = gridW * patchSize;
    const usedH = gridH * patchSize;

    const off = document.createElement("canvas");
    off.width = usedW;
    off.height = usedH;

    const octx = off.getContext("2d");

    // Resize image into the processed canvas.
    // We crop only the leftover strip needed to make dimensions multiples of patchSize.
    octx.drawImage(imgEl, 0, 0, usedW, usedH);

    const N = gridW * gridH;
    const D = patchSize * patchSize * 3;

    const X = new Float32Array(N * D);

    for (let py = 0; py < gridH; py++) {
      for (let px = 0; px < gridW; px++) {
        const patchId = py * gridW + px;

        const data = octx.getImageData(
          px * patchSize,
          py * patchSize,
          patchSize,
          patchSize
        ).data;

        for (let i = 0; i < patchSize * patchSize; i++) {
          X[patchId * D + i * 3 + 0] = data[i * 4 + 0] / 255.0;
          X[patchId * D + i * 3 + 1] = data[i * 4 + 1] / 255.0;
          X[patchId * D + i * 3 + 2] = data[i * 4 + 2] / 255.0;
        }
      }
    }

    return {
      X,
      N,
      D,
      gridW,
      gridH,
      originalW,
      originalH,
      processedW,
      processedH,
      usedW,
      usedH
    };
  }

  // --------------------------------------------------------------------------
  // Canvas sizing
  // --------------------------------------------------------------------------

  function setCanvasSizesFromUsedImage() {
    const maxSide = state.maxDisplaySide;
    const scale = maxSide / Math.max(state.usedW, state.usedH);

    const w = Math.max(1, Math.round(state.usedW * scale));
    const h = Math.max(1, Math.round(state.usedH * scale));

    canvasImg.width = w;
    canvasImg.height = h;

    for (let i = 0; i < state.H; i++) {
      if (headHeatCanvases[i]) {
        headHeatCanvases[i].width = w;
        headHeatCanvases[i].height = h;
      }
      if (headRowCanvases[i]) {
        headRowCanvases[i].width = w;
        headRowCanvases[i].height = h;
      }
    }

    state.canvasW = w;
    state.canvasH = h;
  }

  // Multi-head weights and forward pass  // Multi-head weights and forward pass
  // --------------------------------------------------------------------------

  function initWeights() {
    const D = state.D;
    const dk = state.dk;
    const dv = state.dv;
    const H = state.H;

    state.WQ_T_heads = [];
    state.WK_T_heads = [];
    state.WV_T_heads = [];

    for (let h = 0; h < H; h++) {
      state.WQ_T_heads.push(xavier(D, dk));
      state.WK_T_heads.push(xavier(D, dk));
      state.WV_T_heads.push(xavier(D, dv));
    }

    state.W_O_T = xavier(H * dv, D);
  }

  function runAttention() {
    const { X, N, D, H, dk, dv, WQ_T_heads, WK_T_heads, WV_T_heads, W_O_T } = state;

    state.Q_heads = [];
    state.K_heads = [];
    state.V_heads = [];
    state.S_heads = [];
    state.A_heads = [];
    state.Y_heads = [];

    const headWidth = H * dv;
    const Y_concat = new Float32Array(N * headWidth);

    for (let h = 0; h < H; h++) {
      const Q = matmul(X, WQ_T_heads[h], N, D, dk);
      const K = matmul(X, WK_T_heads[h], N, D, dk);
      const V = matmul(X, WV_T_heads[h], N, D, dv);

      const KT = transpose(K, N, dk);
      const S = matmul(Q, KT, N, dk, N);

      const scale = Math.sqrt(dk);
      for (let i = 0; i < S.length; i++) {
        S[i] /= scale;
      }

      const A = softmaxRows(S, N);
      const Y = matmul(A, V, N, N, dv);

      state.Q_heads[h] = Q;
      state.K_heads[h] = K;
      state.V_heads[h] = V;
      state.S_heads[h] = S;
      state.A_heads[h] = A;
      state.Y_heads[h] = Y;

      for (let i = 0; i < N; i++) {
        Y_concat.set(Y.subarray(i * dv, i * dv + dv), i * headWidth + h * dv);
      }
    }

    const Y_final = matmul(Y_concat, W_O_T, N, headWidth, D);
    state.Y_concat = Y_concat;
    state.Y_final = Y_final;
  }

  // --------------------------------------------------------------------------
  // Drawing helpers
  // --------------------------------------------------------------------------

  function patchRect(patchIdx) {
    const px = patchIdx % state.gridW;
    const py = Math.floor(patchIdx / state.gridW);

    const cellW = state.canvasW / state.gridW;
    const cellH = state.canvasH / state.gridH;

    return {
      x: px * cellW,
      y: py * cellH,
      w: cellW,
      h: cellH,
      px,
      py
    };
  }

  function drawGrid(ctx, alpha) {
    const { gridW, gridH, canvasW, canvasH } = state;

    const cellW = canvasW / gridW;
    const cellH = canvasH / gridH;

    ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
    ctx.lineWidth = 0.5;

    for (let x = 0; x <= gridW; x++) {
      ctx.beginPath();
      ctx.moveTo(x * cellW, 0);
      ctx.lineTo(x * cellW, canvasH);
      ctx.stroke();
    }

    for (let y = 0; y <= gridH; y++) {
      ctx.beginPath();
      ctx.moveTo(0, y * cellH);
      ctx.lineTo(canvasW, y * cellH);
      ctx.stroke();
    }
  }

  function drawSelectedPatch(ctx, patchIdx) {
    if (patchIdx === null) return;

    const r = patchRect(patchIdx);

    ctx.strokeStyle = "#FFD700";
    ctx.lineWidth = 2.5;
    ctx.strokeRect(r.x + 1, r.y + 1, r.w - 2, r.h - 2);

    ctx.fillStyle = "rgba(255,215,0,0.15)";
    ctx.fillRect(r.x + 1, r.y + 1, r.w - 2, r.h - 2);
  }

  function attentionColor(weight, N) {
    const t = Math.max(0, Math.min(1, weight * N));
    const hue = 220 - 180 * t;
    const sat = 85;
    const light = 28 + 42 * t;
    const alpha = 0.20 + 0.70 * Math.pow(t, 0.75);

    return `hsla(${hue.toFixed(1)}, ${sat}%, ${light.toFixed(1)}%, ${alpha.toFixed(3)})`;
  }

  // --------------------------------------------------------------------------
  // Draw input image + patch grid
  // --------------------------------------------------------------------------

  function drawImageGrid() {
    const { img, canvasW, canvasH, selectedPatch } = state;

    ctxImg.clearRect(0, 0, canvasW, canvasH);

    ctxImg.filter = "contrast(1.1) saturate(1.12) brightness(0.96)";
    ctxImg.drawImage(img, 0, 0, canvasW, canvasH);
    ctxImg.filter = "none";

    drawGrid(ctxImg, 0.35);
    drawSelectedPatch(ctxImg, selectedPatch);
  }

  // --------------------------------------------------------------------------
  function renderAllHeadsForPatch(patchIdx) {
    for (let h = 0; h < state.H; h++) {
      drawHeatmapForHead(h, patchIdx, headHeatCtxs[h]);
      drawAttentionRowForHead(h, patchIdx, headRowCtxs[h]);
    }
  }

  function drawAllHeadPlaceholders() {
    for (let h = 0; h < state.H; h++) {
      drawEmptyHeatmapForHead(headHeatCtxs[h]);
      drawEmptyAttentionRowForHead(headRowCtxs[h]);
    }
  }

  function drawHeatmapForHead(headIdx, patchIdx, ctx) {
    const { img, A_heads, N, gridW, gridH, canvasW, canvasH } = state;
    const A = A_heads[headIdx];
    const cellW = canvasW / gridW;
    const cellH = canvasH / gridH;

    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.globalAlpha = 0.35;
    ctx.filter = "contrast(1.1) saturate(1.12) brightness(0.96)";
    ctx.drawImage(img, 0, 0, canvasW, canvasH);
    ctx.filter = "none";
    ctx.globalAlpha = 1.0;
    for (let j = 0; j < N; j++) {
      const w = A[patchIdx * N + j];
      const px = j % gridW;
      const py = Math.floor(j / gridW);
      ctx.fillStyle = attentionColor(w, N);
      ctx.fillRect(px * cellW, py * cellH, cellW, cellH);
    }

    drawGrid(ctx, 0.20);
    drawSelectedPatch(ctx, patchIdx);
  }

  function drawAttentionRowForHead(headIdx, patchIdx, ctx) {
    const { A_heads, N, gridW, gridH, canvasW, canvasH } = state;
    const A = A_heads[headIdx];
    const cellW = canvasW / gridW;
    const cellH = canvasH / gridH;

    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.fillStyle = "#10131f";
    ctx.fillRect(0, 0, canvasW, canvasH);

    for (let j = 0; j < N; j++) {
      const w = A[patchIdx * N + j];
      const px = j % gridW;
      const py = Math.floor(j / gridW);
      ctx.fillStyle = attentionColor(w, N);
      ctx.fillRect(px * cellW, py * cellH, cellW, cellH);
    }

    drawGrid(ctx, 0.18);
    drawSelectedPatch(ctx, patchIdx);
  }

  function drawEmptyHeatmapForHead(ctx) {
    ctx.clearRect(0, 0, state.canvasW, state.canvasH);
    ctx.fillStyle = "#1a1d2e";
    ctx.fillRect(0, 0, state.canvasW, state.canvasH);
    ctx.fillStyle = "#555";
    ctx.font = "13px Georgia";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Click a patch ?", state.canvasW / 2, state.canvasH / 2);
  }

  function drawEmptyAttentionRowForHead(ctx) {
    ctx.clearRect(0, 0, state.canvasW, state.canvasH);
    ctx.fillStyle = "#1a1d2e";
    ctx.fillRect(0, 0, state.canvasW, state.canvasH);
    ctx.fillStyle = "#555";
    ctx.font = "13px Georgia";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Selected row of A", state.canvasW / 2, state.canvasH / 2 - 10);
    ctx.fillText("Click a patch", state.canvasW / 2, state.canvasH / 2 + 12);
  }

  // Info panel: attention contributions of selected patch  // Info panel: attention contributions of selected patch
  // --------------------------------------------------------------------------

  function updateInfo(patchIdx) {
    const { A_heads, N, gridW, dk, dv, D, H } = state;
    const px = patchIdx % gridW;
    const py = Math.floor(patchIdx / gridW);
    const blocks = [];

    for (let h = 0; h < H; h++) {
      const A = A_heads[h];
      const row = A.slice(patchIdx * N, (patchIdx + 1) * N);
      const sorted = Array.from(row).map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v);
      const top3 = sorted.slice(0, 3);
      const entropy = -row.reduce((s, v) => s + (v > 1e-12 ? v * Math.log(v) : 0), 0);
      const softness = ((entropy / Math.log(N)) * 100).toFixed(1);
      let softnessLabel = "hard attention";
      if (softness > 70) softnessLabel = "soft attention";

      blocks.push(`
        <div style="background:#151826;border:1px solid #262b40;border-radius:8px;padding:10px 12px;">
          <div style="color:#ddd;font-weight:600;margin-bottom:4px;">Head ${h + 1}</div>
          ${top3.map((t, i) => {
            const tx = t.i % gridW;
            const ty = Math.floor(t.i / gridW);
            return `<div style="color:#bbb;">${i + 1}. patch ${t.i} (row ${ty}, col ${tx}) - weight <strong style="color:#FFD700;">${(t.v * 100).toFixed(2)}%</strong></div>`;
          }).join("")}
          <div style="margin-top:6px;color:#888;">Entropy: ${entropy.toFixed(3)} &nbsp;(${softness}% of max - ${softnessLabel})</div>
        </div>
      `);
    }

    infoPanel.innerHTML = `
      <strong style="color:#e0e0e0;">Patch ${patchIdx}</strong>
      <span style="color:#888;">(row ${py}, col ${px})</span><br>
      <span style="color:#999;">
        D = ${D}
        &nbsp;|&nbsp;
        H = ${H}
        &nbsp;|&nbsp;
        d<sub>k</sub> = ${dk}
        &nbsp;|&nbsp;
        d<sub>v</sub> = ${dv}
        &nbsp;|&nbsp;
        N = ${N} patches
      </span>
      <br><br>
      <strong style="color:#aaa;">Top attended patches:</strong><br>
      <div style="display:grid; gap:10px; margin-top:8px;">${blocks.join("")}</div>
      <br>
      <span style="color:#666; font-size:12px;">Entropy = 0 -> hard attention. Entropy = log(N) = ${Math.log(N).toFixed(2)} -> uniform attention.</span>
    `;
  }

  // Metadata  // Metadata
  // --------------------------------------------------------------------------

  function updateImageMeta() {
    const {
      originalW,
      originalH,
      processedW,
      processedH,
      usedW,
      usedH,
      gridW,
      gridH,
      patchSize,
      N,
      D,
      H,
      dk,
      dv
    } = state;

    if (!imageMeta) return;
    imageMeta.innerHTML = `
      <span style="color:#aaa;">Original image:</span>
      ${originalW} x ${originalH}px
      &nbsp;|&nbsp;
      <span style="color:#aaa;">Demo image:</span>
      ${processedW} x ${processedH}px
      &nbsp;|&nbsp;
      <span style="color:#aaa;">Used region:</span>
      ${usedW} x ${usedH}px
      <br>
      <span style="color:#aaa;">Patch size:</span>
      ${patchSize} x ${patchSize}
      &nbsp;|&nbsp;
      <span style="color:#aaa;">Grid:</span>
      ${gridW} x ${gridH}
      &nbsp;|&nbsp;
      <span style="color:#aaa;">N:</span>
      ${N}
      &nbsp;|&nbsp;
      <span style="color:#aaa;">D:</span>
      ${D}
      &nbsp;|&nbsp;
      <span style="color:#aaa;">H:</span>
      ${H}
      &nbsp;|&nbsp;
      <span style="color:#aaa;">d<sub>k</sub>:</span>
      ${dk}
      &nbsp;|&nbsp;
      <span style="color:#aaa;">d<sub>v</sub>:</span>
      ${dv}
    `;
  }

  // --------------------------------------------------------------------------
  // Render
  // --------------------------------------------------------------------------

  function render() {
    drawImageGrid();
    syncHeadCardVisibility();

    if (state.selectedPatch !== null) {
      renderAllHeadsForPatch(state.selectedPatch);
      updateInfo(state.selectedPatch);
    } else {
      drawAllHeadPlaceholders();
    }
  }

  // Setup image  // Setup image
  // --------------------------------------------------------------------------

  function setupImage(imgEl) {
    try {
      state.img = imgEl;
      state.patchSize = parseInt(patchSizeSelect.value, 10);
      state.H = parseInt(numHeadsSelect.value, 10);
      state.dk = parseInt(dkSelect.value, 10);
      state.dv = state.dk;

      const extracted = extractPatches(imgEl, state.patchSize);

      state.X = extracted.X;
      state.N = extracted.N;
      state.D = extracted.D;

      state.gridW = extracted.gridW;
      state.gridH = extracted.gridH;

      state.originalW = extracted.originalW;
      state.originalH = extracted.originalH;

      state.processedW = extracted.processedW;
      state.processedH = extracted.processedH;

      state.usedW = extracted.usedW;
      state.usedH = extracted.usedH;

      syncHeadCardVisibility();
      setCanvasSizesFromUsedImage();

      initWeights();
      runAttention();

      state.selectedPatch = null;

      updateImageMeta();
      render();

      infoPanel.innerHTML = `
        <em>
          Image loaded: ${state.N} patches
          (${state.gridW}x${state.gridH} grid),
          D = ${state.D},
          H = ${state.H},
          d<sub>k</sub> = ${state.dk},
          d<sub>v</sub> = ${state.dv}.
          Click any patch to see its attention weights.
        </em>
      `;
    } catch (err) {
      infoPanel.innerHTML = `
        <span style="color:#f66;">
          ${err.message}
        </span>
      `;
    }
  }

  // Default synthetic satellite-like image  // Default synthetic satellite-like image
  // --------------------------------------------------------------------------

  function loadDefault() {
    const size = 224;

    const off = document.createElement("canvas");
    off.width = size;
    off.height = size;

    const octx = off.getContext("2d");

    // Sky
    const sky = octx.createLinearGradient(0, 0, 0, size * 0.4);
    sky.addColorStop(0, "#4a90d9");
    sky.addColorStop(1, "#87ceeb");

    octx.fillStyle = sky;
    octx.fillRect(0, 0, size, size * 0.4);

    // Water
    octx.fillStyle = "#1a5f7a";
    octx.fillRect(0, size * 0.4, size, size * 0.35);

    // Water texture
    for (let i = 0; i < 20; i++) {
      octx.strokeStyle = `rgba(255,255,255,${0.05 + Math.random() * 0.08})`;
      octx.lineWidth = 1 + Math.random() * 2;

      octx.beginPath();
      octx.moveTo(
        Math.random() * size,
        size * 0.4 + Math.random() * size * 0.35
      );
      octx.lineTo(
        Math.random() * size,
        size * 0.4 + Math.random() * size * 0.35
      );
      octx.stroke();
    }

    // Land
    octx.fillStyle = "#5a7a42";
    octx.fillRect(0, size * 0.75, size, size * 0.25);

    // Fields
    const fieldColors = ["#4a6a32", "#6a8a52", "#3d5c28", "#7a9a62"];

    for (let i = 0; i < 8; i++) {
      octx.fillStyle = fieldColors[i % fieldColors.length];

      octx.fillRect(
        i * (size / 8),
        size * 0.75,
        size / 8,
        size * 0.25
      );
    }

    // Boat
    octx.fillStyle = "#ffffff";
    octx.fillRect(size * 0.45, size * 0.52, size * 0.08, size * 0.03);

    octx.fillStyle = "#cccccc";
    octx.fillRect(size * 0.47, size * 0.48, size * 0.02, size * 0.04);

    // Shoreline
    octx.strokeStyle = "#8fa06a";
    octx.lineWidth = 3;

    octx.beginPath();
    octx.moveTo(0, size * 0.75);
    octx.lineTo(size, size * 0.75);
    octx.stroke();

    const img = new Image();

    img.onload = function () {
      setupImage(img);
    };

    img.src = off.toDataURL();
  }

  // --------------------------------------------------------------------------
  // Event handlers
  // --------------------------------------------------------------------------

  uploadInput.addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (ev) {
      const img = new Image();
      img.onload = function () {
        setupImage(img);
      };
      img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
  });

  patchSizeSelect.addEventListener("change", function () {
    if (state.img) setupImage(state.img);
  });

  numHeadsSelect.addEventListener("change", function () {
    if (!state.img) return;
    state.H = parseInt(numHeadsSelect.value, 10);
    state.dv = state.dk;
    syncHeadCardVisibility();
    setCanvasSizesFromUsedImage();
    initWeights();
    runAttention();
    state.selectedPatch = null;
    updateImageMeta();
    render();
    infoPanel.innerHTML = `
      <em>H changed to ${state.H}. New random weights were initialized. Click any patch to see its attention weights.</em>
    `;
  });

  dkSelect.addEventListener("change", function () {
    if (!state.img) return;
    state.dk = parseInt(dkSelect.value, 10);
    state.dv = state.dk;
    initWeights();
    runAttention();
    state.selectedPatch = null;
    updateImageMeta();
    render();
    infoPanel.innerHTML = `
      <em>d<sub>k</sub> changed to ${state.dk}. New random weights were initialized. Click any patch to see its attention weights.</em>
    `;
  });

  rerandomButton.addEventListener("click", function () {
    if (!state.img) return;
    initWeights();
    runAttention();
    state.selectedPatch = null;
    render();
    infoPanel.innerHTML = `
      <em>New random weights initialized. Click any patch to see its attention weights.</em>
    `;
  });

  canvasImg.addEventListener("click", function (e) {
    if (!state.img) return;
    const rect = canvasImg.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const cellW = canvasImg.width / state.gridW;
    const cellH = canvasImg.height / state.gridH;
    const gx = Math.floor(x / cellW);
    const gy = Math.floor(y / cellH);
    if (gx >= 0 && gx < state.gridW && gy >= 0 && gy < state.gridH) {
      state.selectedPatch = gy * state.gridW + gx;
      render();
    }
  });

  // --------------------------------------------------------------------------
  // Boot
  // --------------------------------------------------------------------------

  syncHeadCardVisibility();
  if (!loadPersistedUploadIfPresent()) {
    loadDefault();
  }
})();
