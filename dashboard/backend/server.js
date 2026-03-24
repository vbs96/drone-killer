const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const { randomUUID } = require("crypto");

const app = express();
const PORT = 8001;

app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");

  if (req.method === "OPTIONS") {
    return res.sendStatus(204);
  }

  return next();
});

const uploadDir = path.join(__dirname, "uploads");
fs.mkdirSync(uploadDir, { recursive: true });

// Serve saved uploads (audio or image snippets) for frontend previews.
app.use("/uploads", express.static(uploadDir));

const eventsStore = {
  byId: {},
  allIds: [],
};

function getAllEvents() {
  return eventsStore.allIds.map((id) => eventsStore.byId[id]);
}

function getEventTimestampMs(eventRecord) {
  const receivedAtMs = Date.parse(eventRecord.receivedAt);
  if (!Number.isNaN(receivedAtMs)) {
    return receivedAtMs;
  }

  const metadataTimestampMs = Date.parse(eventRecord?.metadata?.timestamp);
  if (!Number.isNaN(metadataTimestampMs)) {
    return metadataTimestampMs;
  }

  return null;
}

const storage = multer.diskStorage({
  destination: function (_req, _file, cb) {
    cb(null, uploadDir);
  },
  filename: function (_req, file, cb) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const safeOriginal = (file.originalname || "event.wav").replace(/[^a-zA-Z0-9._-]/g, "_");
    cb(null, `${timestamp}_${safeOriginal}`);
  },
});

const upload = multer({ storage });

app.post(
  "/events",
  upload.fields([
    { name: "audio", maxCount: 1 },
    { name: "snippet", maxCount: 1 },
  ]),
  (req, res) => {
    const metadataRaw = req.body.metadata;

    if (!metadataRaw) {
      return res.status(400).json({
        ok: false,
        error: "missing metadata field",
      });
    }

    let metadata;
    try {
      metadata = JSON.parse(metadataRaw);
    } catch (err) {
      return res.status(400).json({
        ok: false,
        error: "invalid metadata json",
      });
    }

    const uploadedAudio = req.files?.audio?.[0] || null;
    const uploadedSnippet = req.files?.snippet?.[0] || null;
    const uploadedFile = uploadedSnippet || uploadedAudio;

    const eventRecord = {
      id: randomUUID(),
      receivedAt: new Date().toISOString(),
      metadata,
      audioPath: uploadedAudio ? `/uploads/${uploadedAudio.filename}` : null,
      snippetPath: uploadedSnippet ? `/uploads/${uploadedSnippet.filename}` : null,
      uploadPath: uploadedFile ? `/uploads/${uploadedFile.filename}` : null,
    };

    eventsStore.byId[eventRecord.id] = eventRecord;
    eventsStore.allIds.push(eventRecord.id);

    console.log(JSON.stringify(eventRecord, null, 2));

    return res.json({
      ok: true,
      message: "event received",
      event: eventRecord,
    });
  });

app.get("/events", (req, res) => {
  const lastMinutesRaw = req.query.lastMinutes;
  const allEvents = getAllEvents();

  if (lastMinutesRaw === undefined) {
    return res.json({
      ok: true,
      count: allEvents.length,
      events: allEvents,
    });
  }

  const lastMinutes = Number(lastMinutesRaw);
  if (!Number.isFinite(lastMinutes) || lastMinutes < 0) {
    return res.status(400).json({
      ok: false,
      error: "lastMinutes must be a non-negative number",
    });
  }

  const cutoffMs = Date.now() - lastMinutes * 60 * 1000;
  const filteredEvents = allEvents.filter((eventRecord) => {
    const eventMs = getEventTimestampMs(eventRecord);
    return eventMs !== null && eventMs >= cutoffMs;
  });

  return res.json({
    ok: true,
    count: filteredEvents.length,
    events: filteredEvents,
  });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server listening at http://0.0.0.0:${PORT}`);
});