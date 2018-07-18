package processing;

import processing.core.PApplet;

public class Practice extends PApplet{

	public Practice() {
		// TODO Auto-generated constructor stub
	}
	
	@Override
	public void settings() {
		this.size(800, 600);
	}
	
	@Override
	public void setup() {
	}
	
	public void draw() {
		this.background(0,0,255);
		this.point(100,100);
	}

	public static void main(String[] args) {
		PApplet.main("processing.Practice");
	}

}
