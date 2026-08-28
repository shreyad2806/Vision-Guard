# ---------------------------------------------------------------------------
# VisionGuard Frontend — Multi-stage Dockerfile
# Stage 1: Build the React/Vite app
# Stage 2: Serve with nginx + proxy API to backend
# ---------------------------------------------------------------------------

# ---- Stage 1: Build ----
FROM node:22-alpine AS build

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm ci

# Copy source and build
COPY . .

# Build with API base URL set to "/" so nginx proxies handle routing
ARG VITE_API_BASE_URL=/
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build

# ---- Stage 2: Serve ----
FROM nginx:alpine AS production

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
