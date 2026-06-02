import axios from "axios";

const BASE = "/api";

const getToken = () => localStorage.getItem("mir_token");

const client = axios.create({ baseURL: BASE });

client.interceptors.request.use((config) => {
  const token = getToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

client.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem("mir_token");
      window.location.reload();
    }
    return Promise.reject(err);
  }
);

export const login = (password) =>
  axios.post(`${BASE}/auth/login`, { password }).then((r) => r.data);

export const getImages = (page = 1, pageSize = 50, classFilter = "") =>
  client.get("/images", {
    params: { page, page_size: pageSize, ...(classFilter && { class_filter: classFilter }) },
  }).then((r) => r.data);

export const getClasses = () =>
  client.get("/classes").then((r) => r.data);

export const search = (queryIndex, descriptors, measure, topK,
                        useReranking = false, poolPercent = 25) =>
  client.post("/search", {
    query_index:   queryIndex,
    descriptors,
    measure,
    top_k:         topK,
    use_reranking: useReranking,
    pool_percent:  poolPercent,
  }).then((r) => r.data);

export const getIndexingMetrics = () =>
  client.get("/indexing-metrics").then((r) => r.data);

export const computeMap = (descriptors, measure, topK, maxQueries = 46) =>
  client.post("/map", {
    descriptors,
    measure,
    top_k:        topK,
    max_queries:  maxQueries,
  }).then((r) => r.data);

export const getClipImages = (page = 1, pageSize = 30) =>
  client.get("/clip/images", { params: { page, page_size: pageSize } }).then((r) => r.data);

export const clipTextToImage = (query, topK = 10) =>
  client.post("/clip/text-to-image", { query, top_k: topK }).then((r) => r.data);

export const clipImageToText = (imageIdx, topK = 10) =>
  client.post("/clip/image-to-text", { image_idx: imageIdx, top_k: topK }).then((r) => r.data);

export const clipEvaluate = (imageIndices, textQueries, topK = 10) =>
  client.post("/clip/evaluate", {
    image_indices: imageIndices,
    text_queries:  textQueries,
    top_k:         topK,
  }).then((r) => r.data);
