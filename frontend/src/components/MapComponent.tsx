// frontend/src/components/MapComponent.tsx
"use client";

import { MapContainer, TileLayer, Polygon, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";

// Helper to flip coords just in case
const flipCoords = (coords: any[]) => {
  return coords.map(c => [c[0], c[1]] as [number, number]); 
};

export default function MapComponent({ polygons }: { polygons: any[] }) {
  const center: [number, number] = [12.9716, 77.5946];

  return (
    <MapContainer 
      center={center} 
      zoom={12} 
      style={{ height: "100%", width: "100%", background: "#020617" }}
      zoomControl={false}
    >
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        attribution='&copy; <a href="https://carto.com/">CARTO</a>'
      />
      
      {polygons.map((poly) => {
        const isHigh = poly.risk === "High";
        const color = isHigh ? "#ef4444" : "#f97316";
        
        return (
          <Polygon 
            key={poly.id}
            positions={flipCoords(poly.coordinates)}
            pathOptions={{ color, fillColor: color, fillOpacity: isHigh ? 0.6 : 0.3, weight: 1 }}
          >
            <Popup className="custom-popup">
              <div className="p-1 min-w-[200px] text-slate-900">
                <h3 className={`font-bold text-lg mb-1 ${isHigh ? 'text-red-600' : 'text-orange-500'}`}>
                  {isHigh ? '🚨 HIGH RISK' : '⚠️ MEDIUM RISK'} ({poly.probability}%)
                </h3>
                <hr className="my-2 border-slate-200" />
                
                {poly.primary_violation !== "None" && (
                  <div className="mb-3">
                    <div className="text-xs font-semibold text-slate-500 uppercase">Most Likely Cause</div>
                    <div className="font-bold text-slate-800">{poly.primary_violation}</div>
                    <div className="text-xs text-slate-500">Accounts for {poly.primary_violation_pct}% of historical incidents</div>
                  </div>
                )}
                
                {poly.reasons && poly.reasons.length > 0 && (
                  <div>
                    <div className="text-xs font-semibold text-slate-500 uppercase mb-1">AI Reasoning</div>
                    <ul className="text-sm text-slate-700 list-disc pl-4 space-y-1">
                      {poly.reasons.map((r: string, i: number) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </Popup>
          </Polygon>
        );
      })}
    </MapContainer>
  );
}
