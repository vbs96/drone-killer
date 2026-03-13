import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { App } from './app';
import { EventsService } from './events.service';

const eventsServiceMock: Pick<EventsService, 'pollEvents'> = {
  pollEvents: () => of([]),
};

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: EventsService, useValue: eventsServiceMock }],
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('should expose tileserver style url', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app.tileStyleUrl).toContain('localhost:8081/styles/basic-preview/style.json');
  });

  it('should expose events endpoint url', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app.eventsUrl).toContain('localhost:8001/events');
  });
});
