"use client";

import { useState } from "react";
import { UploadCloud, FileImage, ShieldAlert, Zap, Loader2, IndianRupee, AlertTriangle, CheckCircle2 } from "lucide-react";

type Violation = {
  class_name: string;
  confidence: number;
  fine_inr: number;
  plate_text?: string;
  severity: string;
};

type AnalysisResult = {
  total_violations: number;
  total_fine_inr: number;
  inference_time_ms: number;
  annotated_image: string;
  violations: Violation[];
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setPreviewUrl(URL.createObjectURL(selected));
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("https://autoviolate-cv-intelligent-traffic.onrender.com/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Failed to analyze image.");

      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Something went wrong.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <main className="min-h-screen py-12 px-6">
      <div className="max-w-6xl mx-auto space-y-12">
        {/* Header Section */}
        <header className="text-center space-y-4">
          <div className="inline-flex items-center justify-center p-3 glass-panel rounded-full mb-4">
            <ShieldAlert className="w-8 h-8 text-[#e94560]" />
          </div>
          <h1 className="text-5xl font-extrabold tracking-tight text-white text-glow">
            AutoViolate<span className="text-[#e94560]">-CV</span>
          </h1>
          <p className="text-lg text-gray-400 max-w-2xl mx-auto">
            Next-gen AI enforcement. Upload traffic footage to instantly detect infractions, classify severity, and extract license plates.
          </p>
        </header>

        <div className="grid lg:grid-cols-2 gap-8 items-start">
          {/* Upload Section */}
          <div className="glass-panel p-8 space-y-6 flex flex-col h-full">
            <div className="flex items-center space-x-3 text-xl font-semibold text-white">
              <UploadCloud className="w-6 h-6 text-[#00d4ff]" />
              <h2>Upload Evidence</h2>
            </div>
            
            <label className="flex-1 min-h-[300px] flex flex-col items-center justify-center border-2 border-dashed border-gray-600 rounded-xl hover:border-[#e94560] transition-colors cursor-pointer bg-black/20 group relative overflow-hidden">
              <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
              {previewUrl ? (
                <img src={previewUrl} alt="Preview" className="absolute inset-0 w-full h-full object-contain p-2" />
              ) : (
                <div className="text-center p-6">
                  <FileImage className="w-12 h-12 mx-auto text-gray-500 group-hover:text-[#e94560] transition-colors mb-4" />
                  <p className="text-gray-300 font-medium text-lg">Click to select an image</p>
                  <p className="text-sm text-gray-500 mt-2">Supports JPG, PNG (Max 5MB)</p>
                </div>
              )}
            </label>

            <button
              onClick={handleUpload}
              disabled={!file || isUploading}
              className={`w-full py-4 rounded-xl font-bold text-lg transition-all duration-300 flex items-center justify-center space-x-2
                ${!file || isUploading ? 'bg-gray-800 text-gray-500 cursor-not-allowed' : 'bg-gradient-to-r from-[#e94560] to-[#b32b47] hover:shadow-[0_0_20px_rgba(233,69,96,0.5)] text-white'}`}
            >
              {isUploading ? (
                <>
                  <Loader2 className="w-6 h-6 animate-spin" />
                  <span>Analyzing Scene...</span>
                </>
              ) : (
                <>
                  <Zap className="w-6 h-6" />
                  <span>Run Analysis</span>
                </>
              )}
            </button>
            {error && <p className="text-red-400 text-center font-medium bg-red-400/10 py-3 rounded-lg">{error}</p>}
          </div>

          {/* Results Section */}
          <div className="glass-panel p-8 space-y-6 h-full flex flex-col">
            <div className="flex items-center space-x-3 text-xl font-semibold text-white">
              <AlertTriangle className="w-6 h-6 text-yellow-500" />
              <h2>Analysis Report</h2>
            </div>

            {!result && !isUploading && (
              <div className="flex-1 flex flex-col items-center justify-center text-gray-500 space-y-4">
                <ShieldAlert className="w-16 h-16 opacity-20" />
                <p className="text-lg">Waiting for input...</p>
              </div>
            )}

            {isUploading && (
              <div className="flex-1 flex flex-col items-center justify-center text-[#e94560] space-y-4">
                <Loader2 className="w-16 h-16 animate-spin" />
                <p className="text-lg animate-pulse font-medium">Running YOLOv8 + EfficientNet Pipeline...</p>
              </div>
            )}

            {result && (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                {/* Stats Row */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-white/5 border border-white/10 rounded-xl p-4 text-center">
                    <p className="text-gray-400 text-sm mb-1 uppercase tracking-wider font-semibold">Violations</p>
                    <p className="text-3xl font-bold text-white">{result.total_violations}</p>
                  </div>
                  <div className="bg-white/5 border border-white/10 rounded-xl p-4 text-center">
                    <p className="text-gray-400 text-sm mb-1 uppercase tracking-wider font-semibold">Total Fines</p>
                    <div className="flex items-center justify-center text-[#0f9b58]">
                      <IndianRupee className="w-6 h-6 mr-1" />
                      <p className="text-3xl font-bold">{result.total_fine_inr.toLocaleString()}</p>
                    </div>
                  </div>
                </div>

                {/* Processed Image */}
                <div className="relative rounded-xl overflow-hidden border border-white/10 shadow-2xl">
                  <img src={result.annotated_image} alt="Annotated" className="w-full h-auto object-cover" />
                  <div className="absolute top-3 left-3 bg-black/60 backdrop-blur-md px-3 py-1 rounded-full text-xs font-medium text-white flex items-center border border-white/20">
                    <CheckCircle2 className="w-3 h-3 mr-1.5 text-green-400" />
                    Processed in {result.inference_time_ms.toFixed(1)}ms
                  </div>
                </div>

                {/* Violation List */}
                {result.violations.length > 0 && (
                  <div className="space-y-3">
                    <h3 className="text-gray-300 font-semibold mb-2">Detected Infractions:</h3>
                    {result.violations.map((v, i) => (
                      <div key={i} className="flex items-center justify-between bg-black/30 border border-white/5 p-3 rounded-lg hover:border-white/20 transition-colors">
                        <div className="flex items-center space-x-3">
                          <div className={`w-2 h-2 rounded-full ${v.severity === 'critical' ? 'bg-red-500' : v.severity === 'high' ? 'bg-orange-500' : 'bg-yellow-500'}`} />
                          <div>
                            <p className="text-white font-medium capitalize">{v.class_name.replace(/_/g, ' ')}</p>
                            {v.plate_text && (
                              <p className="text-xs text-gray-400 font-mono mt-0.5 border border-gray-700 inline-block px-1.5 py-0.5 rounded bg-black/50">
                                {v.plate_text}
                              </p>
                            )}
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-bold text-[#0f9b58]">₹{v.fine_inr}</p>
                          <p className="text-xs text-gray-500">{(v.confidence * 100).toFixed(1)}% conf</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
