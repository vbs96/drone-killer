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

// Serve saved audio files so marker popups can preview them in the frontend.
app.use("/uploads", express.static(uploadDir));

const eventsStore = {
  byId: {},
  allIds: [],
};

function getAllEvents() {
  return eventsStore.allIds.map((id) => eventsStore.byId[id]);
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

app.post("/events", upload.single("audio"), (req, res) => {
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

  const eventRecord = {
    id: randomUUID(),
    receivedAt: new Date().toISOString(),
    metadata,
    audioPath: req.file ? `/uploads/${req.file.filename}` : null,
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

app.get("/events", (_req, res) => {
  const events = getAllEvents();
  return res.json({
    ok: true,
    count: events.length,
    events,
  });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server listening at http://0.0.0.0:${PORT}`);
});