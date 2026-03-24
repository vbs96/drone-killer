import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { map, Observable, switchMap, timer } from 'rxjs';

export interface BackendEvent {
  id: string;
  receivedAt: string;
  metadata: any;
  audioPath: string | null;
  snippetPath: string | null;
  uploadPath: string | null;
}

interface EventsResponse {
  ok: boolean;
  count: number;
  events: BackendEvent[];
}

@Injectable({ providedIn: 'root' })
export class EventsService {
  readonly backendBaseUrl = 'http://localhost:8001';
  readonly eventsUrl = 'http://localhost:8001/events';

  constructor(private readonly http: HttpClient) {}

  getEvents(): Observable<BackendEvent[]> {
    return this.http.get<EventsResponse>(this.eventsUrl).pipe(
      map((response) => (Array.isArray(response.events) ? response.events : []))
    );
  }

  pollEvents(intervalMs = 3000): Observable<BackendEvent[]> {
    return timer(0, intervalMs).pipe(switchMap(() => this.getEvents()));
  }

  toAbsoluteBackendUrl(urlPath: string | null): string | null {
    if (!urlPath) {
      return null;
    }

    if (/^https?:\/\//i.test(urlPath)) {
      return urlPath;
    }

    const normalizedPath = urlPath.startsWith('/') ? urlPath : `/${urlPath}`;
    return `${this.backendBaseUrl}${normalizedPath}`;
  }
}
