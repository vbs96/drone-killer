import { AfterViewInit, Component, ElementRef, OnDestroy, ViewChild } from '@angular/core';
import maplibregl, { Map as MapLibreMap, Marker } from 'maplibre-gl';
import { Subscription } from 'rxjs';
import { BackendEvent, EventsService } from './events.service';

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
  eventCount = 0;

  private readonly markersByEventId = new Map<string, Marker>();
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

    this.eventsSubscription = this.eventsService.pollEvents(3000).subscribe({
      next: (events) => this.syncMarkers(events),
      error: (error: unknown) => {
        console.error('Failed to fetch events:', error);
      },
    });
  }

  ngOnDestroy(): void {
    this.eventsSubscription?.unsubscribe();
    for (const marker of this.markersByEventId.values()) {
      marker.remove();
    }
    this.markersByEventId.clear();
    this.map?.remove();
  }

  private syncMarkers(events: BackendEvent[]): void {
    if (!this.map) {
      return;
    }

    this.eventCount = events.length;
    const incomingIds = new Set<string>();

    for (const eventRecord of events) {
      incomingIds.add(eventRecord.id);
      if (this.markersByEventId.has(eventRecord.id)) {
        continue;
      }

      const coordinates = [eventRecord.metadata.gps.lon, eventRecord.metadata.gps.lat] as [number, number] | null;
      console.log(eventRecord.metadata)

      if (!coordinates) {
        continue;
      }

      const marker = new maplibregl.Marker({ color: '#ff6a3d' })
        .setLngLat(coordinates)
        .setPopup(
          new maplibregl.Popup({ offset: 24 }).setText(
            `Event ${eventRecord.id.slice(0, 8)}\n${eventRecord.receivedAt}`
          )
        )
        .addTo(this.map);

      this.markersByEventId.set(eventRecord.id, marker);
    }

    for (const [eventId, marker] of this.markersByEventId.entries()) {
      if (incomingIds.has(eventId)) {
        continue;
      }

      marker.remove();
      this.markersByEventId.delete(eventId);
    }
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
