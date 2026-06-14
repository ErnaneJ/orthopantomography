export interface Detection {
  id: number;
  class_name: string;
  score: number;
  box_x1: number; box_y1: number; box_x2: number; box_y2: number;
  category: string;
  source: string;
  is_valid: number;
}

export interface Report {
  id: number;
  version: number;
  content: string;
  model: string;
  created_at: string;
}

export interface Annotation {
  id: number;
  kind: string;
  content: string | null;
  class_name: string | null;
  box_x1: number | null; box_y1: number | null;
  box_x2: number | null; box_y2: number | null;
  created_at: string;
}

export interface AnalysisListItem {
  id: number;
  filename: string;
  status: string;
  created_at: string;
  image_url: string;
}

export interface AnalysisDetail extends AnalysisListItem {
  error_msg: string | null;
  annotated_url: string | null;
  detections: Detection[];
  reports: Report[];
  user_annotations: Annotation[];
}
