// Closed-loop lamp thermal control
// Target: T_ref = T_out + 5.0 C
// PID with anti-windup: freeze integrator while output is saturated at 100%

const int PIN_PWM_HEATER = 5;
const float DT_S = 1.0f;
const unsigned long SAMPLE_MS = 1000;

// Start with conservative gains; tune in lab.
float Kp = 0.068f;
float Ki = 0.0025f;
float Kd = 0.0f;

float integralTerm = 0.0f;
float prevError = 0.0f;
unsigned long lastTick = 0;

float readTinC() {
  // Replace with your internal temperature sensor reading.
  return 30.0f;
}

float readToutC() {
  // Replace with your external/environment temperature sensor reading.
  return 25.0f;
}

void setup() {
  pinMode(PIN_PWM_HEATER, OUTPUT);
  analogWrite(PIN_PWM_HEATER, 0);
  Serial.begin(115200);
}

void loop() {
  unsigned long now = millis();
  if (now - lastTick < SAMPLE_MS) {
    return;
  }
  lastTick = now;

  float Tin = readTinC();
  float Tout = readToutC();
  float Tref = Tout + 5.0f;

  float error = Tref - Tin;
  float dError = (error - prevError) / DT_S;

  float iCandidate = integralTerm + Ki * error * DT_S;
  float uUnsat = Kp * error + iCandidate + Kd * dError;

  // Saturation in normalized command [0, 1]
  float uSat = uUnsat;
  if (uSat > 1.0f) uSat = 1.0f;
  if (uSat < 0.0f) uSat = 0.0f;

  // Anti-windup by conditional integration.
  // Mandatory behavior requested: stop integration at 100% output.
  bool atUpperSat = (uSat >= 1.0f) && (error > 0.0f);
  bool atLowerSat = (uSat <= 0.0f) && (error < 0.0f);

  if (!atUpperSat && !atLowerSat) {
    integralTerm = iCandidate;
  }

  float u = Kp * error + integralTerm + Kd * dError;
  if (u > 1.0f) u = 1.0f;
  if (u < 0.0f) u = 0.0f;

  int pwm = (int)(255.0f * u + 0.5f);
  analogWrite(PIN_PWM_HEATER, pwm);

  prevError = error;

  Serial.print("Tout="); Serial.print(Tout, 2);
  Serial.print(", Tref="); Serial.print(Tref, 2);
  Serial.print(", Tin="); Serial.print(Tin, 2);
  Serial.print(", e="); Serial.print(error, 2);
  Serial.print(", u=%"); Serial.println(100.0f * u, 1);
}
