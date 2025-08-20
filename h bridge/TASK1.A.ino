int Q1=13;
int Q3=11;
int Q2=12;
int Q4=10;

void setup() {
pinMode(Q1,OUTPUT);
pinMode(Q2,OUTPUT);
pinMode(Q3,OUTPUT);
pinMode(Q4,OUTPUT);
 
}

void loop() {
  digitalWrite(Q1,HIGH);
  digitalWrite(Q4,HIGH);
  digitalWrite(Q2,LOW);
  digitalWrite(Q3,LOW);
  delay(5000);
  
  digitalWrite(Q2,HIGH);
  digitalWrite(Q3,HIGH);
  digitalWrite(Q1,LOW);
  digitalWrite(Q4,LOW);
  delay(5000);}
