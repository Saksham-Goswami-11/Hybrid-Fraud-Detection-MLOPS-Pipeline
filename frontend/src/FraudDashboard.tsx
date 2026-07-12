import React, { useState, useEffect } from "react";
import {
  ShieldAlert,
  Activity,
  DollarSign,
  Clock,
  Sliders,
  Cpu,
  RefreshCw,
  Info,
  Server,
  Zap,
  CheckCircle,
  AlertTriangle,
} from "lucide-react";

// ==========================================
// Mock Initial Data & Fallbacks
// ==========================================
const DEFAULT_INPUTS = {
  Time: 86400,
  Amount: 1250.0,
  V1: -1.35,
  V2: 0.07,
  V3: 2.53,
  V4: 1.37,
  V5: -0.33,
  V10: 0.09,
  V11: -0.55,
  V12: -0.61,
  V14: -0.31,
  V17: 0.2,
  V18: 0.02,
  V21: -0.01,
  V27: 0.13,
};

const MOCK_FRAUD_RESPONSE = {
  is_fraud: true,
  fraud_probability: 0.867,
  threshold: 0.3104,
  shap_values: {
    Amount: 0.45,
    V17: 0.32,
    V12: -0.15,
    V1: 0.08,
    V2: -0.04,
  },
  top_risk_factors: [
    "Amount (+0.4500)",
    "V17 (+0.3200)",
    "V12 (-0.1500)",
    "V1 (+0.0800)",
    "V2 (-0.0400)",
  ],
};

const MOCK_SAFE_RESPONSE = {
  is_fraud: false,
  fraud_probability: 0.021,
  threshold: 0.3104,
  shap_values: {
    Amount: -0.12,
    V17: -0.08,
    V12: -0.22,
    V1: 0.02,
    V2: 0.01,
  },
  top_risk_factors: [
    "V12 (-0.2200)",
    "Amount (-0.1200)",
    "V17 (-0.0800)",
    "V1 (+0.0200)",
    "V2 (+0.0100)",
  ],
};

const OFFLINE_SAMPLE_DATA: Record<number, Record<string, number>> = {
  2: {
    Time: 145248.0, Amount: 50.0,
    V1: 1.9140268216, V2: -0.4900679879, V3: -0.3261113125, V4: 0.6047107392, V5: -0.8501359998,
    V6: -0.736318677, V7: -0.5240579625, V8: -0.0886141066, V9: 1.0911251047, V10: 0.0934843578,
    V11: -0.8923046259, V12: 0.0272205159, V13: -0.2437902096, V14: 0.0317740067, V15: 0.9006238971,
    V16: 0.5360321616, V17: -0.6484080941, V18: 0.18307234, V19: -0.4863224942, V20: -0.1395787634,
    V21: 0.2109584289, V22: 0.6393378791, V23: 0.147522552, V24: 0.0736542664, V25: -0.3183782466,
    V26: 0.3506122627, V27: -0.0238434747, V28: -0.0371393315
  },
  1870: {
    Time: 146022.0, Amount: 1.18,
    V1: 0.9086366582, V2: 2.8490240149, V3: -5.6473429634, V4: 6.0094147781, V5: 0.216656395,
    V6: -2.397014424, V7: -1.8193078859, V8: 0.3385269877, V9: -2.819882773, V10: -4.0630981083,
    V11: 2.9411900927, V12: -6.1513621909, V13: -1.9895285353, V14: -9.1509510056, V15: -0.6042899988,
    V16: -1.9522903998, V17: -2.8925553322, V18: -0.9120579607, V19: -1.5637399671, V20: 0.2419212947,
    V21: 0.4072604613, V22: -0.3974348525, V23: -0.0800058534, V24: -0.1685965452, V25: 0.4650584978,
    V26: 0.2105097562, V27: 0.648704799, V28: 0.3602243303
  },
  6861: {
    Time: 148074.0, Amount: 0.0,
    V1: -2.2192186022, V2: 0.7278314111, V3: -5.4582299465, V4: 5.9248498471, V5: 3.9324638238,
    V6: -3.0859842366, V7: -1.6778699877, V8: 0.8650746104, V9: -3.1772602889, V10: -3.4192073841,
    V11: 3.6931739422, V12: -3.9784397551, V13: -1.7185908746, V14: -8.6362973937, V15: -0.2429648215,
    V16: 1.1748841732, V17: 2.134606357, V18: 2.594364833, V19: -1.2575889799, V20: 0.9647718037,
    V21: 0.417471746, V22: -0.8173433841, V23: -0.028752402, V24: 0.0257225109, V25: -0.8258353432,
    V26: -0.0130890305, V27: 0.4132911887, V28: -0.1313873464
  }
};

export default function FraudDashboard() {
  // Input form state (initialize as strings to allow typing negative signs, decimals, etc.)
  const [inputs, setInputs] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {};
    Object.entries(DEFAULT_INPUTS).forEach(([key, val]) => {
      initial[key] = val.toString();
    });
    return initial;
  });

  // API response and UI states
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [, setError] = useState<string | null>(null);

  // MLOps metadata state
  const [systemHealth, setSystemHealth] = useState({
    status: "healthy",
    modelName: "RandomForestClassifier",
    threshold: 0.3104,
    latency: "< 45ms",
    isLive: false,
  });

  // Test set samples list
  const [samples, setSamples] = useState<{ index: number; label: string }[]>([]);
  const [selectedSample, setSelectedSample] = useState<string>("");
  const [customIndexInput, setCustomIndexInput] = useState<string>("");
  const [cardId, setCardId] = useState<string>("card_analyst_demo");
  const [sampleLoading, setSampleLoading] = useState(false);
  const [sampleError, setSampleError] = useState<string | null>(null);

  // Fetch sample options on mount
  useEffect(() => {
    fetch("http://localhost:8000/sample-transactions")
      .then((res) => {
        if (!res.ok) throw new Error("Could not fetch samples");
        return res.json();
      })
      .then((data) => setSamples(data))
      .catch((err) => {
        console.warn("Backend samples offline. Using offline sample list.", err);
        // Fallback offline sample list matching backend presets
        setSamples([
          { index: 2, label: "Legitimate Transaction #1 ($50.00) — Row 2" },
          { index: 3, label: "Legitimate Transaction #2 ($14.95) — Row 3" },
          { index: 4, label: "Legitimate Transaction #3 ($7.70) — Row 4" },
          { index: 5, label: "Legitimate Transaction #4 ($6.99) — Row 5" },
          { index: 1870, label: "Fraudulent Transaction #1 ($1.18) — Row 1870" },
          { index: 1884, label: "Fraudulent Transaction #2 ($2.22) — Row 1884" },
          { index: 2235, label: "Fraudulent Transaction #3 ($0.77) — Row 2235" },
          { index: 2635, label: "Fraudulent Transaction #4 ($94.82) — Row 2635" },
          { index: 4135, label: "Fraudulent Transaction #5 ($8.00) — Row 4135" },
          { index: 6861, label: "Fraudulent Transaction #6 ($0.00) — Row 6861" },
        ]);
      });
  }, []);

  // Fetch specific sample features and populate form
  const loadSample = async (index: number) => {
    setSampleLoading(true);
    setSampleError(null);
    try {
      const response = await fetch(`http://localhost:8000/sample-transaction/${index}`);
      if (!response.ok) {
        throw new Error(`Failed to load transaction at row ${index}`);
      }
      const data = await response.json();
      
      // Update inputs with string representation of the retrieved columns
      const updated: Record<string, string> = {};
      Object.entries(data).forEach(([key, val]) => {
        updated[key] = (val as any).toString();
      });

      setInputs(updated);
      setResult(null); // Clear previous results to prompt user to run scoring
    } catch (err: any) {
      console.warn("Backend lookup failed, trying local fallback values...");
      if (OFFLINE_SAMPLE_DATA[index]) {
        const data = OFFLINE_SAMPLE_DATA[index];
        const updated: Record<string, string> = {};
        Object.entries(data).forEach(([key, val]) => {
          updated[key] = val.toString();
        });
        setInputs(updated);
        setResult(null);
      } else {
        setSampleError(`Could not load row ${index} (offline fallback only supports rows 2, 1870, and 6861). Please ensure the FastAPI backend is running.`);
      }
    } finally {
      setSampleLoading(false);
    }
  };

  // Check backend health on mount
  useEffect(() => {
    fetch("http://localhost:8000/health")
      .then((res) => res.json())
      .then((data) => {
        setSystemHealth((prev) => ({
          ...prev,
          status: data.status || "healthy",
          modelName: data.model_type || "RandomForestClassifier",
          isLive: true,
        }));
      })
      .catch(() => {
        // Fallback to mock live status if API is down
        setSystemHealth((prev) => ({ ...prev, isLive: false }));
      });
  }, []);

  // Handle form field change
  const handleInputChange = (field: string, value: string) => {
    setInputs((prev) => ({ ...prev, [field]: value }));
  };

  // Submit scoring request
  const analyzeTransaction = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Build the request payload (ensure all PCA features are provided, defaulting the rest to 0.0)
    const payload: Record<string, any> = {
      card_id: cardId || "card_default_123",
    };
    for (let i = 1; i <= 28; i++) {
      payload[`V${i}`] = 0.0;
    }
    Object.entries(inputs).forEach(([key, val]) => {
      payload[key] = parseFloat(val) || 0.0;
    });

    try {
      const startTime = performance.now();
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`API returned status ${response.status}`);
      }

      const data = await response.json();
      const endTime = performance.now();

      // Normalize different api response variations
      const parsedResult = {
        is_fraud: data.is_fraud !== undefined ? data.is_fraud : data.prediction === 1,
        fraud_probability: data.fraud_probability !== undefined ? data.fraud_probability : data.probability,
        threshold: data.threshold || 0.3104,
        shap_values: data.shap_values || {},
        top_risk_factors: data.top_risk_factors || [],
        rule_triggered: data.rule_triggered || false,
        rule_reasons: data.rule_reasons || [],
      };

      setResult(parsedResult);
      setSystemHealth((prev) => ({
        ...prev,
        latency: `${Math.round(endTime - startTime)}ms`,
      }));
    } catch (err) {
      console.warn("Backend API offline. Falling back to local model simulation...");
      // Simulate backend behavior locally for demonstration / offline use
      await new Promise((resolve) => setTimeout(resolve, 800)); // Smooth loading transition

      const amountNum = parseFloat(inputs.Amount) || 0;
      const v17Num = parseFloat(inputs.V17) || 0;
      const v12Num = parseFloat(inputs.V12) || 0;
      const v1Num = parseFloat(inputs.V1) || 0;
      const v2Num = parseFloat(inputs.V2) || 0;

      const isMockFraud = amountNum > 1000 || v17Num > 1.5;
      const mockData = isMockFraud ? MOCK_FRAUD_RESPONSE : MOCK_SAFE_RESPONSE;

      setResult({
        ...mockData,
        rule_triggered: false,
        rule_reasons: [],
        // Interpolate input values to make SHAP values look reactive to user changes
        shap_values: {
          Amount: amountNum > 1000 ? 0.45 : -0.12,
          V17: v17Num > 1.0 ? 0.32 : -0.08,
          V12: v12Num < -0.5 ? 0.25 : -0.15,
          V1: v1Num * -0.05,
          V2: v2Num * 0.03,
        },
      });

      setSystemHealth((prev) => ({
        ...prev,
        latency: "< 15ms (local simulation)",
      }));
    } finally {
      setLoading(false);
    }
  };

  // Convert shap values to chart items
  const getShapChartData = () => {
    if (!result) return [];
    return Object.entries(result.shap_values)
      .map(([feature, val]: [string, any]) => ({
        name: feature,
        value: val,
        percentage: Math.min(Math.abs(val) * 100, 100),
        effect: val >= 0 ? "increases risk" : "decreases risk",
      }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans p-6 md:p-12">
      {/* Container */}
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center pb-6 border-b border-slate-800 gap-4">
          <div>
            <div className="flex items-center gap-3">
              <span className="p-2 bg-rose-500/10 rounded-lg text-rose-500">
                <ShieldAlert className="w-6 h-6 animate-pulse" />
              </span>
              <h1 className="text-2xl font-bold tracking-tight text-white">
                Aegis Risk Terminal
              </h1>
            </div>
            <p className="text-sm text-slate-400 mt-1">
              Enterprise Machine Learning Transaction Scoring & Compliance Explainer
            </p>
          </div>

          {/* MLOps Health Ribbon */}
          <div className="flex flex-wrap gap-3 bg-slate-900 border border-slate-800/80 rounded-xl p-3 text-xs">
            <div className="flex items-center gap-2 px-3 py-1 bg-slate-950 rounded-lg border border-slate-800">
              <Server className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-slate-400">Model:</span>
              <span className="font-semibold text-white">{systemHealth.modelName}</span>
            </div>
            
            <div className="flex items-center gap-2 px-3 py-1 bg-slate-950 rounded-lg border border-slate-800">
              <Activity className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-slate-400">Status:</span>
              <span className="flex items-center gap-1.5 font-semibold text-white">
                <span className={`w-2 h-2 rounded-full ${systemHealth.isLive ? "bg-emerald-500" : "bg-amber-500"}`}></span>
                {systemHealth.isLive ? "Live API" : "Simulated Model"}
              </span>
            </div>

            <div className="flex items-center gap-2 px-3 py-1 bg-slate-950 rounded-lg border border-slate-800">
              <Sliders className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-slate-400">Threshold:</span>
              <span className="font-mono font-semibold text-white">{systemHealth.threshold}</span>
            </div>

            <div className="flex items-center gap-2 px-3 py-1 bg-slate-950 rounded-lg border border-slate-800">
              <Zap className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-slate-400">Latency:</span>
              <span className="font-mono font-semibold text-white">{systemHealth.latency}</span>
            </div>
          </div>
        </header>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Transaction Input Form */}
          <section className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Cpu className="w-5 h-5 text-indigo-400" />
                Scoring Engine Inputs
              </h2>
              <button 
                onClick={() => {
                  const reset: Record<string, string> = {};
                  Object.entries(DEFAULT_INPUTS).forEach(([k, v]) => {
                    reset[k] = v.toString();
                  });
                  setInputs(reset);
                  setSelectedSample("");
                  setCustomIndexInput("");
                  setCardId("card_analyst_demo");
                }}
                className="text-xs text-slate-400 hover:text-white flex items-center gap-1 bg-slate-950 px-2 py-1.5 rounded-lg border border-slate-850 hover:bg-slate-800 transition"
              >
                <RefreshCw className="w-3 h-3" /> Reset
              </button>
            </div>

            {/* Quick Sample Loader */}
            <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-350 flex items-center gap-1.5">
                  <Sliders className="w-3.5 h-3.5 text-indigo-400" />
                  Load Test Set Transaction
                </span>
                {sampleLoading && (
                  <RefreshCw className="w-3.5 h-3.5 text-indigo-400 animate-spin" />
                )}
              </div>

              {/* Dropdown Preset Selector */}
              <div className="space-y-2">
                <select
                  value={selectedSample}
                  onChange={(e) => {
                    setSelectedSample(e.target.value);
                    if (e.target.value !== "") {
                      loadSample(parseInt(e.target.value));
                    }
                  }}
                  className="w-full bg-slate-900 border border-slate-850 rounded-lg px-3 py-2.5 text-xs text-slate-300 focus:outline-none focus:ring-1 focus:ring-indigo-500/50 focus:border-indigo-500 transition animate-none"
                >
                  <option value="">-- Choose Preset Test Sample (No Label) --</option>
                  {samples.map((s) => (
                    <option key={s.index} value={s.index}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Row Index Lookup Input */}
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="Or enter row index (0 - 56961)"
                  value={customIndexInput}
                  onChange={(e) => setCustomIndexInput(e.target.value)}
                  className="flex-1 bg-slate-900 border border-slate-850 rounded-lg px-3 py-2 text-xs text-white font-mono placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500/50 focus:border-indigo-500"
                />
                <button
                  type="button"
                  onClick={() => {
                    const idx = parseInt(customIndexInput.trim());
                    if (!isNaN(idx)) {
                      loadSample(idx);
                      setSelectedSample("");
                    }
                  }}
                  disabled={sampleLoading || !customIndexInput.trim()}
                  className="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-500 rounded-lg text-xs font-semibold text-white transition active:translate-y-[1px]"
                >
                  Load Row
                </button>
              </div>

              {sampleError && (
                <p className="text-[10px] text-rose-450 font-medium leading-tight">
                  ⚠️ {sampleError}
                </p>
              )}
            </div>

            <form onSubmit={analyzeTransaction} className="space-y-5">
              {/* Primary Fields */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider flex items-center gap-1">
                    <DollarSign className="w-3.5 h-3.5 text-slate-500" /> Amount (USD)
                  </label>
                  <input
                    type="text"
                    value={inputs.Amount}
                    onChange={(e) => handleInputChange("Amount", e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-white font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition"
                    placeholder="150.00"
                    required
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider flex items-center gap-1">
                    <Clock className="w-3.5 h-3.5 text-slate-500" /> Time Index (s)
                  </label>
                  <input
                    type="text"
                    value={inputs.Time}
                    onChange={(e) => handleInputChange("Time", e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-white font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition"
                    placeholder="0"
                    required
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider flex items-center gap-1">
                    <Sliders className="w-3.5 h-3.5 text-slate-500" /> Card ID (Track)
                  </label>
                  <input
                    type="text"
                    value={cardId}
                    onChange={(e) => setCardId(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-white font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition"
                    placeholder="card_123"
                    required
                  />
                </div>
              </div>

              <div className="text-[10px] text-slate-400 bg-slate-950/40 border border-slate-800/50 rounded-lg p-2.5 leading-normal flex items-start gap-1.5 font-mono">
                <span>💡</span>
                <span>
                  <strong>Tip:</strong> Click the scoring button 4 times with the same Card ID in rapid succession to trigger the behavioral velocity limit rule.
                </span>
              </div>

              {/* PCA Latent Features Heading */}
              <div className="pt-2">
                <div className="flex items-center gap-2 mb-3">
                  <div className="h-[1px] bg-slate-800 flex-1"></div>
                  <span className="text-[10px] uppercase font-bold text-slate-500 tracking-widest px-2 bg-slate-900">
                    PCA Dimension Inputs
                  </span>
                  <div className="h-[1px] bg-slate-800 flex-1"></div>
                </div>

                {/* PCA Inputs Grid */}
                <div className="grid grid-cols-3 gap-3">
                  {Object.keys(DEFAULT_INPUTS)
                    .filter((key) => key !== "Amount" && key !== "Time")
                    .map((key) => (
                      <div key={key}>
                        <label className="block text-[10px] font-mono font-semibold text-slate-400 mb-1">
                          {key}
                        </label>
                        <input
                          type="text"
                          value={inputs[key]}
                          onChange={(e) => handleInputChange(key, e.target.value)}
                          className="w-full bg-slate-950 border border-slate-800/80 rounded-lg px-2 py-1.5 text-xs text-white font-mono focus:outline-none focus:ring-1 focus:ring-indigo-500/50 focus:border-indigo-500"
                        />
                      </div>
                    ))}
                </div>
              </div>

              {/* Action Button */}
              <button
                type="submit"
                disabled={loading}
                className={`w-full py-3.5 rounded-xl font-semibold text-white tracking-wide transition flex items-center justify-center gap-2.5 ${
                  loading
                    ? "bg-indigo-650/50 cursor-not-allowed text-slate-350"
                    : "bg-indigo-600 hover:bg-indigo-500 shadow-lg shadow-indigo-600/20 active:translate-y-[1px]"
                }`}
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    Running Model Inferencing...
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4 fill-white" />
                    Run Fraud Scoring
                  </>
                )}
              </button>
            </form>
          </section>

          {/* Right Column: Prediction Results & Explanations */}
          <div className="lg:col-span-7 space-y-6">
            
            {/* Conditional Result Banner */}
            {!result && !loading && (
              <div className="h-full min-h-[400px] flex flex-col justify-center items-center text-center p-8 bg-slate-900/50 border border-dashed border-slate-800 rounded-2xl">
                <div className="p-4 bg-slate-900 border border-slate-800 rounded-full text-slate-400 mb-4">
                  <Sliders className="w-8 h-8" />
                </div>
                <h3 className="text-lg font-semibold text-white">No Scoring Task Run</h3>
                <p className="text-sm text-slate-400 mt-1 max-w-sm">
                  Adjust dimension inputs and run the fraud scoring engine to populate model classification outputs.
                </p>
              </div>
            )}

            {loading && (
              <div className="h-full min-h-[400px] flex flex-col justify-center items-center text-center p-8 bg-slate-900 border border-slate-800 rounded-2xl">
                <div className="relative w-16 h-16 flex items-center justify-center">
                  <div className="absolute inset-0 border-4 border-indigo-500/20 rounded-full"></div>
                  <div className="absolute inset-0 border-4 border-t-indigo-500 rounded-full animate-spin"></div>
                </div>
                <h3 className="text-lg font-semibold text-white mt-4">Predictive Evaluation</h3>
                <p className="text-sm text-slate-400 mt-1">
                  Computing feature contributions and local SHAP explanation values...
                </p>
              </div>
            )}

            {result && !loading && (
              <div className="space-y-6 animate-fade-in">
                
                {/* Result Card Banner */}
                <div
                  className={`border rounded-2xl p-6 shadow-xl relative overflow-hidden transition-all duration-350 ${
                    result.is_fraud
                      ? (result.rule_triggered ? "bg-amber-950/20 border-amber-500/40 shadow-amber-950/10" : "bg-rose-950/20 border-rose-500/40 shadow-rose-950/10")
                      : "bg-emerald-950/20 border-emerald-500/40 shadow-emerald-950/10"
                  }`}
                >
                  <div className="flex items-start justify-between relative z-10 gap-4">
                    <div>
                      <span
                        className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider mb-3 ${
                          result.is_fraud
                            ? (result.rule_triggered ? "bg-amber-500/10 text-amber-400 border border-amber-500/30" : "bg-rose-500/10 text-rose-400")
                            : "bg-emerald-500/10 text-emerald-400"
                        }`}
                      >
                        {result.is_fraud ? (
                          result.rule_triggered ? (
                            <>
                              <AlertTriangle className="w-3.5 h-3.5" />
                              Rule Triggered
                            </>
                          ) : (
                            <>
                              <AlertTriangle className="w-3.5 h-3.5" />
                              Fraud Alert
                            </>
                          )
                        ) : (
                          <>
                            <CheckCircle className="w-3.5 h-3.5" />
                            Approved
                          </>
                        )}
                      </span>
                      <h3 className="text-2xl font-black text-white uppercase tracking-tight">
                        {result.is_fraud 
                          ? (result.rule_triggered ? "SUSPICIOUS (RULE ESCALATION)" : "FRAUD SUSPECTED") 
                          : "TRANSACTION APPROVED"}
                      </h3>
                      <p className="text-sm text-slate-400 mt-1">
                        {result.is_fraud
                          ? (result.rule_triggered 
                             ? `Security Rule Escalation: ${result.rule_reasons.join(", ")}`
                             : "Model score exceeds risk threshold. Flagging for manual analyst review.")
                          : "Model score within safe boundaries. Automated clearance granted."}
                      </p>
                    </div>

                    {/* Big Score Ring */}
                    <div className="text-right">
                      <div className="text-xs uppercase font-semibold text-slate-400 tracking-wider">
                        Risk Probability
                      </div>
                      <div
                        className={`text-4xl font-extrabold mt-1 font-mono ${
                          result.is_fraud 
                            ? (result.rule_triggered ? "text-amber-500" : "text-rose-500") 
                            : "text-emerald-500"
                        }`}
                      >
                        {(result.fraud_probability * 100).toFixed(2)}%
                      </div>
                      <div className="text-[10px] text-slate-500 font-mono mt-0.5">
                        Threshold: {result.threshold}
                      </div>
                    </div>
                  </div>

                  {/* Backdrop Glow */}
                  <div
                    className={`absolute w-36 h-36 rounded-full blur-[80px] -bottom-12 -right-12 ${
                      result.is_fraud 
                        ? (result.rule_triggered ? "bg-amber-500/15" : "bg-rose-500/15")
                        : "bg-emerald-500/15"
                    }`}
                  ></div>
                </div>

                {/* Explainability & SHAP Bar Chart */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
                  <div>
                    <h3 className="text-md font-semibold text-white flex items-center gap-2">
                      <Sliders className="w-4 h-4 text-indigo-400" />
                      SHAP Explainability & Risk Attribution
                    </h3>
                    <p className="text-xs text-slate-400 mt-1">
                      Attribution weights pushing risk scoring toward or away from decision threshold.
                    </p>
                  </div>

                  {/* Attributions Progress Stack */}
                  <div className="space-y-4">
                    {getShapChartData().map((item) => (
                      <div key={item.name} className="space-y-1.5">
                        <div className="flex justify-between text-xs font-mono">
                          <span className="font-medium text-slate-200">{item.name}</span>
                          <span
                            className={
                              item.value >= 0 ? "text-rose-400 font-bold" : "text-emerald-400 font-bold"
                            }
                          >
                            {item.value >= 0 ? "+" : ""}
                            {item.value.toFixed(4)} ({item.effect === "increases risk" ? "Risk Push" : "Mitigative"})
                          </span>
                        </div>
                        {/* Progress Bar Container */}
                        <div className="h-3 bg-slate-950 rounded-full overflow-hidden flex">
                          {item.value >= 0 ? (
                            // Risk Push - Left-aligned red bar
                            <div
                              style={{ width: `${item.percentage}%` }}
                              className="h-full bg-gradient-to-r from-indigo-500/30 to-rose-500 rounded-full"
                            ></div>
                          ) : (
                            // Mitigative - Left-aligned green bar
                            <div
                              style={{ width: `${item.percentage}%` }}
                              className="h-full bg-gradient-to-r from-indigo-500/30 to-emerald-500 rounded-full"
                            ></div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Compliance Explainer Note */}
                  <div className="p-4 bg-slate-950 border border-slate-800/80 rounded-xl space-y-2">
                    <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-300">
                      <Info className="w-4 h-4 text-indigo-400" />
                      Compliance Narrative Explainer
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed font-mono">
                      This transaction was assessed by model version{" "}
                      <span className="text-white font-semibold">{systemHealth.modelName}</span>. 
                      {result.is_fraud ? (
                        <>
                          {" "}Attribution weights reveal the flag was heavily driven by{" "}
                          <span className="text-rose-400 font-semibold">
                            {getShapChartData().filter(item => item.value > 0).map(item => item.name).slice(0, 2).join(" & ")}
                          </span>, pushing probability to { (result.fraud_probability * 100).toFixed(1) }% (above the threshold of {result.threshold}).
                        </>
                      ) : (
                        <>
                          {" "}Attribution weights show significant risk reduction from{" "}
                          <span className="text-emerald-400 font-semibold">
                            {getShapChartData().filter(item => item.value < 0).map(item => item.name).slice(0, 2).join(" & ")}
                          </span>, keeping the probability at safe { (result.fraud_probability * 100).toFixed(1) }% (below the threshold of {result.threshold}).
                        </>
                      )}
                    </p>
                  </div>

                </div>

              </div>
            )}

          </div>

        </div>

      </div>
    </div>
  );
}
