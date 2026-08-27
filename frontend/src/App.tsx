import { BrowserRouter, Routes, Route } from "react-router-dom";
import AppLayout from "./components/layout/AppLayout";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import AnalysisDetail from "./pages/AnalysisDetail";

export default function App() {
  return (
    <BrowserRouter>
      <AppLayout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/history" element={<History />} />
          <Route path="/history/:id" element={<AnalysisDetail />} />
        </Routes>
      </AppLayout>
    </BrowserRouter>
  );
}
