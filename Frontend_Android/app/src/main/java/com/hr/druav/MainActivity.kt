package com.hr.druav

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.NavController
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.NavigationUI
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController
import com.google.firebase.database.DataSnapshot
import com.google.firebase.database.DatabaseError
import com.google.firebase.database.FirebaseDatabase
import com.google.firebase.database.ValueEventListener
import com.hr.druav.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var appBarConfiguration: AppBarConfiguration
    private lateinit var binding: ActivityMainBinding
    private lateinit var navController: NavController

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Set up the toolbar
        setSupportActionBar(binding.appBarMain.toolbar)

        // Initialize Navigation Controller
        navController = findNavController(R.id.nav_host_fragment_content_main)

        // Configure AppBar with top-level destinations (excluding the maintenance prediction item)
        appBarConfiguration = AppBarConfiguration(
            setOf(R.id.nav_scanner, R.id.nav_network, R.id.nav_intelligence),
            binding.drawerLayout
        )

        // Set up ActionBar with NavController and the NavigationView
        setupActionBarWithNavController(navController, appBarConfiguration)
        binding.navView.setupWithNavController(navController)
        binding.navBtmview.setupWithNavController(navController)

        // Handle Floating Action Button Click
        binding.appBarMain.fabVoice.setOnClickListener {
            val intent = Intent(this, BotActivity::class.java)
            startActivity(intent)
        }

        // Manually handle click on "Maintenance Prediction" menu item (id: R.id.n)
        binding.navView.menu.findItem(R.id.n).setOnMenuItemClickListener {
            Log.d("MainActivity", "Maintenance Prediction menu item clicked")
            updateMaintenancePredictionDate()
            // Optionally close the drawer
            binding.drawerLayout.closeDrawers()
            true
        }
    }

    /**
     * Fetches the predicted maintenance date from Firebase and updates the TextView
     * in the navigation header (nav_header_main.xml) with id pred_date.
     */
    private fun updateMaintenancePredictionDate() {
        // Get the header view from the NavigationView
        val headerView = binding.navView.getHeaderView(0)
        val predDateTextView = headerView.findViewById<TextView>(R.id.pred_date)
        Log.d("MainActivity", "Updating maintenance prediction date...")

        // Get a reference to the Firebase node "predictedMaintenanceDate"
        val database = FirebaseDatabase.getInstance()
        val ref = database.getReference("maintenance_schedule").child("next_maintenance_date")

        ref.addListenerForSingleValueEvent(object : ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {
                Log.d("MainActivity", "Firebase snapshot: $snapshot")
                val predictedDate = snapshot.getValue(String::class.java)
                if (predictedDate != null) {
                    Log.d("MainActivity", "Fetched predicted date: $predictedDate")
                    predDateTextView.text = predictedDate
                } else {
                    Log.d("MainActivity", "No predicted date found in Firebase")
                    predDateTextView.text = "No Date Available"
                }
            }


            override fun onCancelled(error: DatabaseError) {
                Log.e("MainActivity", "Error fetching predicted date: ${error.message}")
            }
        })
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.main, menu)
        return true
    }

    override fun onSupportNavigateUp(): Boolean {
        return NavigationUI.navigateUp(navController, appBarConfiguration) || super.onSupportNavigateUp()
    }
}
