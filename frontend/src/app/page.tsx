"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { SlidersHorizontal, Map as MapIcon, ShieldAlert, Clock, Info, Database, UploadCloud, Loader2, ServerCrash } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const MapComponent = dynamic(() => import("../components/MapComponent"), { ssr: false });

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Dashboard() {
  const [dates, setDates] = useState<string[]>([]);
  const [topVios, setTopVios] = useState<string[]>([]);
  
  const [selectedDate, setSelectedDate] = useState<string>("");
  const [selectedHour, setSelectedHour] = useState<number>(12);
  const [userWeights, setUserWeights] = useState<Record<string, number>>({});
  
  const [polygons, setPolygons] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Retraining state
  const [isRetraining, setIsRetraining] = useState(false);
  const [retrainStep, setRetrainStep] = useState("");
  const [retrainError, setRetrainError] = useState("");
  
  const [backendError, setBackendError] = useState(false);

  // Fetch initial config
  useEffect(() => {
    fetch(`${API_URL}/api/config`)
      .then(r => r.json())
      .then(data => {
        setDates(data.dates || []);
        const initialDate = data.dates?.length > 0 ? data.dates[0] : "";
        if(initialDate) setSelectedDate(initialDate);
        
        const vios = data.top_violations || [];
        setTopVios(vios);
        
        const initialWeights: Record<string, number> = {};
        vios.forEach((v: string) => initialWeights[v] = 3.0);
        setUserWeights(initialWeights);
        setLoading(false);
        
        if(data.status && data.status.status === "running") {
          setIsRetraining(true);
          setRetrainStep(data.status.step);
        }
        
        // Fetch initial predictions
        if (initialDate) {
          fetch(`${API_URL}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              date: initialDate,
              hour: 12,
              user_weights: initialWeights
            })
          }).then(r => r.json()).then(d => setPolygons(d.polygons));
        }
      })
      .catch(e => {
        console.error("FastAPI backend not running", e);
        setBackendError(true);
        setLoading(false);
      });
  }, []);

  const fetchPredictions = async () => {
    if(!selectedDate) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          date: selectedDate,
          hour: selectedHour,
          user_weights: userWeights
        })
      });
      const data = await res.json();
      setPolygons(data.polygons);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  // Removed auto-fetch on Date/Hour change. Intelligence is now strictly driven by the Apply button.
  
  const handleApply = () => {
    fetchPredictions();
  }

  // Polling for retrain status
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRetraining) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_URL}/api/retrain/status`);
          const data = await res.json();
          setRetrainStep(data.step);
          if (data.status === "idle" && data.step === "Complete") {
            setIsRetraining(false);
            setRetrainStep("");
            window.location.reload(); // Reload dashboard to fetch fresh models and config
          } else if (data.status === "error") {
            setIsRetraining(false);
            setRetrainError(data.step);
          }
        } catch (e) {
          console.error("Error polling retrain status", e);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [isRetraining]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setIsRetraining(true);
    setRetrainStep("Uploading CSV to Backend...");
    setRetrainError("");

    try {
      const res = await fetch(`${API_URL}/api/retrain`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if(data.message) {
        setRetrainStep("Initializing Pipeline...");
      }
    } catch (err) {
      setIsRetraining(false);
      setRetrainError("Failed to upload file.");
    }
  };

  if (backendError) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center text-slate-400 space-y-6">
        <ServerCrash size={64} className="text-red-500/80" />
        <div className="text-center">
          <h2 className="text-2xl font-bold text-white mb-2">Intelligence Engine Offline</h2>
          <p className="text-sm">The GridLock FastAPI backend is currently unreachable.</p>
          <p className="text-xs text-slate-500 mt-1">Please ensure uvicorn is running on port 8000.</p>
        </div>
        <button 
          onClick={() => window.location.reload()} 
          className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-medium rounded shadow-lg transition-colors"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  if(loading && dates.length === 0) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center text-slate-400 space-y-4">
        <Loader2 size={48} className="text-blue-500 animate-spin" />
        <p className="font-medium animate-pulse">Initializing Intelligence Engine...</p>
      </div>
    );
  }

  const highCount = polygons.filter(p => p.risk === "High").length;
  const medCount = polygons.filter(p => p.risk === "Medium").length;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 flex font-sans overflow-hidden">
      
      {/* Sidebar */}
      <motion.div 
        initial={{ x: -300, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ type: "spring", damping: 20 }}
        className="w-80 bg-slate-900 border-r border-slate-800 p-6 flex flex-col gap-8 h-screen overflow-y-auto shrink-0 z-20 shadow-2xl relative"
      >
        <div>
          <h1 className="text-xl font-bold flex items-center gap-2 mb-2">
            <ShieldAlert className="text-red-500" />
            GridLock AI
          </h1>
          <p className="text-xs text-slate-400">Predictive Enforcement Intelligence</p>
          
          <div className="mt-6 p-3 bg-slate-800/50 border border-slate-700/50 rounded-lg">
            <h2 className="text-xs font-semibold text-slate-300 flex items-center gap-2 mb-2">
              <Database size={14} className="text-blue-400" /> Model Pipeline
            </h2>
            <label className="flex items-center justify-center gap-2 w-full py-2 px-3 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium rounded cursor-pointer transition-colors">
              <UploadCloud size={14} />
              Upload Data & Retrain
              <input type="file" accept=".csv" className="hidden" onChange={handleFileUpload} />
            </label>
            <p className="text-[10px] text-slate-500 mt-2 leading-tight">
              ⚠️ Deployed free-tiers will crash (OOM) during training. Use locally.
            </p>
          </div>
        </div>

        {/* Time Travel */}
        <div className="space-y-4">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-2">
            <Clock size={16} /> Time Travel Engine
          </h2>
          
          <div className="space-y-2">
            <label className="text-xs text-slate-400">Future Date</label>
            <select 
              className="w-full bg-slate-800 border border-slate-700 rounded p-2.5 text-sm focus:ring-2 focus:ring-blue-500 outline-none text-slate-200"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
            >
              {(dates || []).map((d, i) => (
                <option key={d} value={d}>
                  {i === 0 ? "Today (Live Baseline)" : `In ${i} Days`}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-xs text-slate-400 flex justify-between">
              <span>Target Hour</span>
              <span className="font-mono bg-slate-800 px-2 py-0.5 rounded text-blue-400">{selectedHour}:00</span>
            </label>
            <input 
              type="range" min="0" max="23" value={selectedHour}
              onChange={(e) => setSelectedHour(parseInt(e.target.value))}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
          </div>
        </div>

        <div className="h-px bg-slate-800" />

        {/* Commander Panel */}
        <div className="space-y-4 pb-12">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-2">
            <SlidersHorizontal size={16} /> Priority Scaling
          </h2>
          
          <div className="space-y-5">
            {(topVios || []).map(vio => (
              <div key={vio} className="space-y-2">
                <label className="text-xs text-slate-400 flex justify-between items-end">
                  <span className="capitalize">{vio.toLowerCase()}</span>
                  <span className="font-mono text-slate-500">{userWeights[vio]?.toFixed(1)}</span>
                </label>
                <input 
                  type="range" min="0" max="5" step="0.5" 
                  value={userWeights[vio] ?? 3.0}
                  onChange={(e) => setUserWeights({...userWeights, [vio]: parseFloat(e.target.value)})}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-red-500"
                />
              </div>
            ))}
            
            <button 
              onClick={handleApply}
              className="w-full bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white font-medium py-3 rounded-md transition-colors text-sm mt-8 shadow-lg shadow-blue-900/20 uppercase tracking-wide"
            >
              Generate Intelligence
            </button>
          </div>
        </div>

      </motion.div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-screen relative bg-slate-950">
        {/* Top bar metrics */}
        <div className="absolute top-0 left-0 right-0 p-6 flex items-start gap-4 z-[400] pointer-events-none">
          <motion.div initial={{ y: -50, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="bg-slate-900/80 backdrop-blur-md rounded-xl p-4 px-6 border border-slate-800 shadow-2xl pointer-events-auto">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">High Risk Zones</div>
            <div className="text-3xl font-bold text-red-500 flex items-center gap-2">
              {highCount}
              <span className="text-xs bg-red-500/20 text-red-400 px-2 py-1 rounded-full font-medium tracking-normal">Action Required</span>
            </div>
          </motion.div>
          <motion.div initial={{ y: -50, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.1 }} className="bg-slate-900/80 backdrop-blur-md rounded-xl p-4 px-6 border border-slate-800 shadow-2xl pointer-events-auto">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Medium Risk Zones</div>
            <div className="text-3xl font-bold text-orange-500 flex items-center gap-2">
              {medCount}
              <span className="text-xs bg-orange-500/20 text-orange-400 px-2 py-1 rounded-full font-medium tracking-normal">Monitor</span>
            </div>
          </motion.div>
          
          <div className="ml-auto">
            {loading && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-slate-900/80 backdrop-blur-md rounded-full px-4 py-2 border border-slate-800 shadow-2xl flex items-center gap-3 text-slate-300 text-sm">
                <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                Computing intelligence...
              </motion.div>
            )}
          </div>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 relative bg-slate-950 flex flex-col z-10">
          <MapComponent polygons={polygons} topVios={topVios} />
        </div>

        {/* Retraining Modal Overlay */}
        <AnimatePresence>
          {isRetraining && (
            <motion.div 
              initial={{ opacity: 0 }} 
              animate={{ opacity: 1 }} 
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4"
            >
              <motion.div 
                initial={{ scale: 0.95 }}
                animate={{ scale: 1 }}
                className="bg-slate-900 border border-slate-800 p-8 rounded-xl max-w-md w-full shadow-2xl"
              >
                <h2 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
                  <Database className="text-blue-500" />
                  Updating Intelligence
                </h2>
                <p className="text-sm text-slate-400 mb-6">
                  The GridLock pipeline is processing new data, re-calculating historical momenta, and re-training the Random Forest model. This may take a few minutes.
                </p>
                
                {retrainError ? (
                  <div className="p-4 bg-red-500/10 border border-red-500/50 rounded-lg text-red-400 text-sm">
                    <strong>Error:</strong> {retrainError}
                    <button 
                      onClick={() => setIsRetraining(false)}
                      className="mt-4 w-full bg-red-600 hover:bg-red-500 text-white py-2 rounded"
                    >
                      Close
                    </button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-300 font-medium">{retrainStep || "Initializing..."}</span>
                      <Loader2 size={16} className="text-blue-500 animate-spin" />
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div className="bg-blue-500 h-full w-1/2 animate-pulse rounded-full" />
                    </div>
                  </div>
                )}
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
