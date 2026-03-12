FROM node:20-alpine AS build

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

FROM nginx:1.27-alpine

# Replace the default nginx welcome files with Angular browser output.
RUN rm -rf /usr/share/nginx/html/*
COPY --from=build /app/dist/drone-dashboard/browser /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
