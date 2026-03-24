import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';
import { BackendEvent, EventsService } from './events.service';

@Component({
  selector: 'app-events-drawer',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './events-drawer.component.html',
  styleUrl: './events-drawer.component.scss',
})
export class EventsDrawerComponent {
  @Input() events: BackendEvent[] = [];

  constructor(private readonly eventsService: EventsService) {}

  trackByEvent(_: number, eventRecord: BackendEvent): string {
    return `${eventRecord.id}|${eventRecord.receivedAt}`;
  }

  getEventType(eventRecord: BackendEvent): string {
    const metadataType = this.readMetadataType(eventRecord.metadata);
    if (metadataType === 'audio' || metadataType === 'video') {
      return metadataType.toUpperCase();
    }

    if (eventRecord.audioPath) {
      return 'AUDIO';
    }

    if (eventRecord.snippetPath || eventRecord.uploadPath) {
      return 'VIDEO';
    }

    return 'UNKNOWN';
  }

  getSnippetUrl(eventRecord: BackendEvent): string | null {
    return this.eventsService.toAbsoluteBackendUrl(eventRecord.snippetPath ?? eventRecord.uploadPath);
  }

  getAudioUrl(eventRecord: BackendEvent): string | null {
    return this.eventsService.toAbsoluteBackendUrl(eventRecord.audioPath);
  }

  getCoordinatesLabel(eventRecord: BackendEvent): string | null {
    const coordinates = this.extractCoordinates(eventRecord.metadata);
    if (!coordinates) {
      return null;
    }

    return `Lat ${coordinates[1].toFixed(5)}, Lng ${coordinates[0].toFixed(5)}`;
  }

  private readMetadataType(metadata: unknown): string | null {
    if (!metadata || typeof metadata !== 'object') {
      return null;
    }

    const value = (metadata as Record<string, unknown>)['type'];
    if (typeof value !== 'string') {
      return null;
    }

    const normalized = value.trim().toLowerCase();
    return normalized.length > 0 ? normalized : null;
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
