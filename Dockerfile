# --- build stage ---
FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY main.cpp httplib.h ./
COPY src ./src
RUN g++ -std=c++17 -O2 main.cpp -o db -pthread

# --- runtime stage ---
FROM ubuntu:24.04
WORKDIR /app
COPY --from=build /app/db ./db
COPY index.html ./
EXPOSE 8080
CMD ["./db"]
