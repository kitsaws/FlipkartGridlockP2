"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { SlidersHorizontal, Map as MapIcon, ShieldAlert, Clock, Info } from "lucide-react";
import { motion } from "framer-motion";

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

  // Fetch initial config
  useEffect(() => {
    fetch(`${API_URL}/api/config`)
      .then(r => r.json())
      .then(data => {
        setDates(data.dates);
        const initialDate = data.dates.length > 0 ? data.dates[0] : "";
        if(initialDate) setSelectedDate(initialDate);
        setTopVios(data.top_violations);
        
        const initialWeights: Record<string, number> = {};
        data.top_violations.forEach((v: string) => initialWeights[v] = 3.0);
        setUserWeights(initialWeights);
        setLoading(false);
        
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

  if(loading && dates.length === 0) {
    return <div className="min-h-screen bg-slate-950 flex items-center justify-center text-slate-400">Loading Intelligence Engine... Ensure FastAPI is running on port 8000.</div>
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
              {dates.map((d, i) => (
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
            {topVios.map(vio => (
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

        {/* Map Container */}
        <motion.div 
            initial={{ opacity: 0 }} 
            animate={{ opacity: 1 }} 
            transition={{ duration: 1 }}
            className="flex-1 relative z-0"
        >
          <MapComponent polygons={polygons} />
        </motion.div>
      </div>
    </div>
  );
}
