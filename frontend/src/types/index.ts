export type Status = 'queued' | 'running' | 'review_pending' | 'rejected' | 'completed' | 'failed' | 'cancelled'
export type Role = 'admin' | 'editor' | 'reader'
export interface User { id: string; username: string }
export interface Member extends User { role: Role; is_owner: boolean }
export interface Corpus { id: string; name: string; role: Role; created_at: string }
export interface PaperVersion {
  id: string; paper_id: string; title: string; filename: string; sha256: string; status: string
  error?: string; page_count?: number; warnings: string[]; created_at: string; byte_size: number; file_url: string
  folder: string; tags: string[]; is_active: boolean; format: string
}
export interface Snapshot { id: string; corpus_id: string; manifest: Array<{title: string; paper_version_id: string; index_revision_id: string}> }
export interface Evidence {
  id: string; quote: string; title: string; paper_version_id: string; block_id: string; char_start: number; char_end: number; pdf_page_index: number; pdf_page_number: number
  printed_page_label?: string; bbox: number[]; file_url: string; section: string; location_kind: string; location_label: string
}
export interface Claim { id: string; text: string; conditions: string[]; semantic_verified: boolean; evidence: Array<{evidence_id: string; relation: string}> }
export interface Answer { id: string; evidence_status: string; claims: Claim[]; evidence: Evidence[]; gaps: string[]; issues: Array<{type: string; message: string}>; stop_reason: string; clarification?: string; usage: Record<string, number> }
export interface Run { id: string; request_id: string; is_owner: boolean; corpus_id: string; question: string; snapshot_id: string; status: Status; mode: string; attempt: number; config?: {evidence_gate_enabled?: boolean}; error?: string; answer?: Answer; answer_sha256?: string | null; answer_review?: {decision: string; reason?: string} | null; created_at: string }
export interface RunEvent { seq: number; kind: string; attempt: number; payload: {message?: string; [key: string]: unknown}; created_at: string }
export interface HistoryItem {id: string; question: string; status: Status; created_at: string}
export interface Trace {request_id: string; nodes: Array<{node: string; seconds: number; error?: string}>; retrievals: Array<{round: number; query: string; rankings: Array<{chunk_id: string; title: string; score: number}>}>; followup_count: number; error?: string}
export interface Health {database: string; execution: string; diagnostic_mode: boolean}
export interface CodeTestEnvironment {
  image_id: string; command: string[]; network: string; read_only_root: boolean; workspace_read_only: boolean
  memory: string; cpus: string; pids: string; file_manifest_sha256: string
  docker_engine: {endpoint: string; context: string; client_version: string; server_version: string}
}
export interface CodeRepairTask {
  id: string; repository: string; objective: string; base_ref: string; base_commit: string
  selected_files: string[]; status: string; summary?: string; patch_sha256?: string; patch?: string
  test_exit_code?: number; test_output?: string; test_environment?: CodeTestEnvironment
  verification_sha256?: string; workspace_state: string; error?: string; created_at: string
}
export interface RepairExperience {
  id: string; task_id: string; artifact_sha256: string; objective: string; summary: string
  verified_command: string; verified_result: string; created_at: string
}
export interface CodeRepairReview {
  id: string; task_id: string; decision: string; artifact_sha256: string; bundle_sha256?: string
  reason?: string; feedback_category: string; root_cause_layer: string; uncertainty?: string
  reviewer_id: string; created_at: string
}
