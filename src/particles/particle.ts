/**
 * Force matrix: forceMatrix[sourceType][targetType] = force value.
 * Negative = repulsion, positive = attraction.
 * Values typically range from -1.0 to 1.0.
 */
export type ForceMatrix = Readonly<Record<string, Readonly<Record<string, number>>>>;

export abstract class Particle {
  x: number;
  y: number;
  vx = 0;
  vy = 0;

  /** RGB color in 0–1 range */
  abstract color: [number, number, number];

  constructor(x: number, y: number) {
    this.x = x;
    this.y = y;
  }

  /** Wrap position to the opposite edge (toroidal topology) */
  wrapPosition(width: number, height: number) {
    this.x = ((this.x % width) + width) % width;
    this.y = ((this.y % height) + height) % height;
  }
}
