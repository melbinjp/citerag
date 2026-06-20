# ---------------------------------------------------------------------------
# Frontend Dockerfile — multi-stage build
# Stage 1: Node build (Vite/React)
# Stage 2: Nginx static file server
# Req 14.1 (frontend Docker deployment), 14.2 (CPU-only, no GPU)
# ---------------------------------------------------------------------------

# ---- Build stage ----
FROM node:20-alpine AS build
WORKDIR /app

# Install dependencies first (layer-cache friendly)
COPY package*.json ./
RUN npm ci

# Copy source
COPY . .

# VITE_API_URL is baked into the bundle at build time.
# Override via: docker build --build-arg VITE_API_URL=http://your-backend:7860
ARG VITE_API_URL=http://localhost:7860
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# ---- Serve stage ----
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
