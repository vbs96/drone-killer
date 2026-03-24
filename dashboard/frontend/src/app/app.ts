import { AfterViewInit, Component, ElementRef, OnDestroy, ViewChild } from '@angular/core';
import maplibregl, { Map as MapLibreMap, Marker } from 'maplibre-gl';
import { Subscription } from 'rxjs';
import { BackendEvent, EventsService } from './events.service';

type EventType = 'audio' | 'video';

interface EventsAtLocation {
  key: string;
  coordinates: [number, number];
  audioEvent?: BackendEvent;
  videoEvent?: BackendEvent;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App implements AfterViewInit, OnDestroy {
  @ViewChild('mapContainer', { static: true })
  private readonly mapContainer!: ElementRef<HTMLDivElement>;

  readonly tileStyleUrl = 'http://localhost:8081/styles/basic-preview/style.json';
  readonly eventsUrl = 'http://localhost:8001/events';
  readonly lastMinutesOptions = [5, 10, 15, 30, 60];
  selectedLastMinutes = 15;
  eventCount = 0;

  private readonly markersByLocationKey = new Map<string, Marker>();
  private map: MapLibreMap | undefined;
  private eventsSubscription: Subscription | undefined;

  constructor(private readonly eventsService: EventsService) {}

  ngAfterViewInit(): void {
    this.map = new maplibregl.Map({
      container: this.mapContainer.nativeElement,
      style: this.tileStyleUrl,
      center: [25, 45.9],
      zoom: 6,
      maxZoom: 18,
    });

    this.map.addControl(new maplibregl.NavigationControl(), 'top-right');
    this.map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left');

    this.eventsSubscription = this.eventsService
      .pollEvents(3000, () => this.selectedLastMinutes)
      .subscribe({
      next: (events) => this.syncMarkers(events),
      error: (error: unknown) => {
        console.error('Failed to fetch events:', error);
      },
    });
  }

  onLastMinutesChange(rawValue: string): void {
    const parsedValue = Number(rawValue);
    if (Number.isFinite(parsedValue) && parsedValue >= 0) {
      this.selectedLastMinutes = parsedValue;
    }
  }

  ngOnDestroy(): void {
    this.eventsSubscription?.unsubscribe();
    for (const marker of this.markersByLocationKey.values()) {
      marker.remove();
    }
    this.markersByLocationKey.clear();
    this.map?.remove();
  }

  private syncMarkers(events: BackendEvent[]): void {
    if (!this.map) {
      return;
    }

    this.eventCount = events.length;
    const groupedByLocation = this.getLatestEventsByLocation(events);
    const incomingLocationKeys = new Set<string>();

    for (const locationEvents of groupedByLocation.values()) {
      incomingLocationKeys.add(locationEvents.key);

      // Recreate marker in case event composition changed (audio/video/latest timestamp).
      const existingMarker = this.markersByLocationKey.get(locationEvents.key);
      if (existingMarker) {
        existingMarker.remove();
      }

      const marker = new maplibregl.Marker({ element: this.buildMarkerElement(locationEvents) })
        .setLngLat(locationEvents.coordinates)
        .setPopup(new maplibregl.Popup({ offset: 24 }).setDOMContent(this.buildPopupContent(locationEvents)))
        .addTo(this.map);

      this.markersByLocationKey.set(locationEvents.key, marker);
    }

    for (const [locationKey, marker] of this.markersByLocationKey.entries()) {
      if (incomingLocationKeys.has(locationKey)) {
        continue;
      }

      marker.remove();
      this.markersByLocationKey.delete(locationKey);
    }
  }

  private getLatestEventsByLocation(events: BackendEvent[]): Map<string, EventsAtLocation> {
    const groupedByLocation = new Map<string, EventsAtLocation>();

    for (const eventRecord of events) {
      const coordinates = this.extractCoordinates(eventRecord.metadata);
      if (!coordinates) {
        continue;
      }

      const eventType = this.extractEventType(eventRecord);
      if (!eventType) {
        continue;
      }

      const key = this.getCoordinateKey(coordinates);
      const existing = groupedByLocation.get(key) ?? {
        key,
        coordinates,
      };

      if (eventType === 'audio') {
        if (!existing.audioEvent || this.getEventTimestampMs(eventRecord) >= this.getEventTimestampMs(existing.audioEvent)) {
          existing.audioEvent = eventRecord;
        }
      } else if (!existing.videoEvent || this.getEventTimestampMs(eventRecord) >= this.getEventTimestampMs(existing.videoEvent)) {
        existing.videoEvent = eventRecord;
      }

      groupedByLocation.set(key, existing);
    }

    return groupedByLocation;
  }

  private buildMarkerElement(locationEvents: EventsAtLocation): HTMLDivElement {
    const markerElement = document.createElement('div');
    markerElement.className = 'event-marker';

    const hasAudio = Boolean(locationEvents.audioEvent);
    const hasVideo = Boolean(locationEvents.videoEvent);
    if (hasAudio && hasVideo) {
      markerElement.classList.add('event-marker-mixed');
    } else if (hasAudio) {
      markerElement.classList.add('event-marker-audio');
    } else {
      markerElement.classList.add('event-marker-video');
    }

    return markerElement;
  }

  private buildPopupContent(locationEvents: EventsAtLocation): HTMLElement {
    const root = document.createElement('div');
    root.className = 'event-popup';

    const title = document.createElement('strong');
    const hasAudio = Boolean(locationEvents.audioEvent);
    const hasVideo = Boolean(locationEvents.videoEvent);
    if (hasAudio && hasVideo) {
      title.textContent = 'Audio + Video events';
    } else if (hasAudio) {
      title.textContent = 'Audio event';
    } else {
      title.textContent = 'Video event';
    }
    root.appendChild(title);

    if (locationEvents.audioEvent) {
      root.appendChild(this.buildPopupSection('Audio', locationEvents.audioEvent));
    }

    if (locationEvents.videoEvent) {
      root.appendChild(this.buildPopupSection('Video', locationEvents.videoEvent));
    }

    return root;
  }

  private buildPopupSection(label: string, eventRecord: BackendEvent): HTMLElement {
    const section = document.createElement('section');
    section.className = 'event-popup-section';

    const eventTitle = document.createElement('div');
    eventTitle.className = 'event-popup-section-title';
    eventTitle.textContent = `${label} ${eventRecord.id.slice(0, 8)}`;
    section.appendChild(eventTitle);

    const time = document.createElement('div');
    time.textContent = new Date(eventRecord.receivedAt).toLocaleString();
    time.className = 'event-popup-time';
    section.appendChild(time);

    const audioUrl = this.eventsService.toAbsoluteBackendUrl(eventRecord.audioPath);
    const snippetPath = eventRecord.snippetPath ?? eventRecord.uploadPath;
    const snippetUrl = this.eventsService.toAbsoluteBackendUrl(snippetPath);

    if (snippetUrl) {
      const snippetImage = document.createElement('img');
      snippetImage.src = snippetUrl;
      snippetImage.alt = 'Detected drone snippet';
      snippetImage.loading = 'lazy';
      snippetImage.className = 'event-popup-image';
      section.appendChild(snippetImage);
    }

    if (audioUrl) {
      const audio = document.createElement('audio');
      audio.controls = true;
      audio.preload = 'none';
      audio.src = audioUrl;
      audio.className = 'event-popup-audio';
      section.appendChild(audio);
    }

    if (!audioUrl && !snippetUrl) {
      const noMedia = document.createElement('div');
      noMedia.textContent = 'No media uploaded for this event.';
      noMedia.className = 'event-popup-noaudio';
      section.appendChild(noMedia);
    }

    return section;
  }

  private extractEventType(eventRecord: BackendEvent): EventType | null {
    const metadataType = this.readMetadataType(eventRecord.metadata);
    if (metadataType === 'audio' || metadataType === 'video') {
      return metadataType;
    }

    if (eventRecord.audioPath) {
      return 'audio';
    }

    if (eventRecord.snippetPath || eventRecord.uploadPath) {
      return 'video';
    }

    return null;
  }

  private readMetadataType(metadata: unknown): EventType | null {
    if (!metadata || typeof metadata !== 'object') {
      return null;
    }

    const value = (metadata as Record<string, unknown>)['type'];
    if (typeof value !== 'string') {
      return null;
    }

    const normalized = value.trim().toLowerCase();
    if (normalized === 'audio') {
      return 'audio';
    }
    if (normalized === 'video') {
      return 'video';
    }

    return null;
  }

  private getCoordinateKey([lng, lat]: [number, number]): string {
    return `${lng.toFixed(6)},${lat.toFixed(6)}`;
  }

  private getEventTimestampMs(eventRecord: BackendEvent): number {
    const receivedAtMs = Date.parse(eventRecord.receivedAt);
    if (Number.isFinite(receivedAtMs)) {
      return receivedAtMs;
    }

    if (eventRecord.metadata && typeof eventRecord.metadata === 'object') {
      const metadataTimestamp = (eventRecord.metadata as Record<string, unknown>)['timestamp'];
      if (typeof metadataTimestamp === 'string') {
        const metadataTimestampMs = Date.parse(metadataTimestamp);
        if (Number.isFinite(metadataTimestampMs)) {
          return metadataTimestampMs;
        }
      }
    }

    return 0;
  }

  private extractCoordinates(metadata: unknown): [number, number] | null {
    if (!metadata || typeof metadata !== 'object') {
      return null;
    }

    const metadataObject = metadata as Record<string, unknown>;
    const gps = metadataObject['gps'];
    if (gps && typeof gps === 'object') {
      const gpsObject = gps as Record<string, unknown>;
      const lat = this.readNumber(gpsObject, ['lat', 'latitude']);
      const lon = this.readNumber(gpsObject, ['lon', 'lng', 'longitude', 'long']);
      if (lat !== null && lon !== null) {
        return [lon, lat];
      }
    }

    const candidates: Record<string, unknown>[] = [metadataObject];
    const location = metadataObject['location'];
    if (location && typeof location === 'object') {
      candidates.push(location as Record<string, unknown>);
    }

    for (const candidate of candidates) {
      const lat = this.readNumber(candidate, ['lat', 'latitude']);
      const lng = this.readNumber(candidate, ['lng', 'lon', 'longitude', 'long']);
      if (lat !== null && lng !== null) {
        return [lng, lat];
      }

      const coordinates = candidate['coordinates'];
      if (!Array.isArray(coordinates) || coordinates.length < 2) {
        continue;
      }

      const lngFromArray = Number(coordinates[0]);
      const latFromArray = Number(coordinates[1]);
      if (Number.isFinite(lngFromArray) && Number.isFinite(latFromArray)) {
        return [lngFromArray, latFromArray];
      }
    }

    return null;
  }

  private readNumber(source: Record<string, unknown>, keys: string[]): number | null {
    for (const key of keys) {
      const value = source[key];
      if (typeof value === 'number' && Number.isFinite(value)) {
        return value;
      }

      if (typeof value === 'string') {
        const parsed = Number(value);
        if (Number.isFinite(parsed)) {
          return parsed;
        }
      }
    }

    return null;
  }
}
